from __future__ import annotations
import argparse
import os
import json
from typing import List, Tuple
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from tasks.common.dataset import NextFrameDataset


class GRUFeatureExtractor(nn.Module):
    def __init__(self, teams: int, agents: int, hidden: int = 128):
        super().__init__()
        self.teams = teams
        self.agents = agents
        self.in_dim = teams * agents * 2
        self.gru = nn.GRU(self.in_dim, hidden, batch_first=True)
        # do not include a head; we return the last hidden state as features
    def forward(self, x):
        B, W, T, A, D = x.shape
        x = x.reshape(B, W, T * A * D)
        h, _ = self.gru(x)
        return h[:, -1, :]


def collate(batch):
    s = torch.tensor(np.stack([b["state"] for b in batch]), dtype=torch.float32)
    y = torch.tensor(np.array([b.get("intent", 0) for b in batch]), dtype=torch.long)
    return s, y


def to_device(batch, device):
    return batch[0].to(device), batch[1].to(device)


def train_linear_probe(features: np.ndarray, labels: np.ndarray, val: Tuple[np.ndarray, np.ndarray], epochs: int = 30, lr: float = 0.1) -> Tuple[nn.Module, float, float]:
    x = torch.tensor(features, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.long)
    x_val = torch.tensor(val[0], dtype=torch.float32)
    y_val = torch.tensor(val[1], dtype=torch.long)
    num_classes = int(y.max().item()) + 1 if y.numel() > 0 else 1
    clf = nn.Linear(x.shape[1], num_classes)
    opt = torch.optim.Adam(clf.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    for _ in range(epochs):
        opt.zero_grad()
        logits = clf(x)
        loss = loss_fn(logits, y)
        loss.backward()
        opt.step()
    with torch.no_grad():
        train_acc = (clf(x).argmax(dim=1) == y).float().mean().item()
        val_acc = (clf(x_val).argmax(dim=1) == y_val).float().mean().item()
    return clf, train_acc, val_acc


def kmeans(X: np.ndarray, k: int, iters: int = 50, seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    N, D = X.shape
    idx = rng.choice(N, size=k, replace=False)
    C = X[idx]
    for _ in range(iters):
        # assign
        dist = ((X[:, None, :] - C[None, :, :]) ** 2).sum(axis=2)
        z = dist.argmin(axis=1)
        # update
        for j in range(k):
            pts = X[z == j]
            if len(pts) > 0:
                C[j] = pts.mean(axis=0)
    return z, C


def clustering_purity(pred: np.ndarray, labels: np.ndarray) -> float:
    K = int(pred.max()) + 1 if pred.size else 1
    total = len(labels)
    hits = 0
    for k in range(K):
        mask = (pred == k)
        if mask.sum() == 0:
            continue
        lbls, counts = np.unique(labels[mask], return_counts=True)
        hits += int(counts.max())
    return float(hits / max(1, total))


def compute_features(ds: NextFrameDataset, model: GRUFeatureExtractor, device: str, max_items: int | None = None) -> Tuple[np.ndarray, np.ndarray]:
    loader = DataLoader(ds, batch_size=256, shuffle=False, collate_fn=collate)
    feats = []
    labs = []
    model.eval()
    with torch.no_grad():
        count = 0
        for s, y in loader:
            s = s.to(device)
            f = model(s).cpu().numpy()
            feats.append(f)
            labs.append(y.numpy())
            count += s.size(0)
            if max_items is not None and count >= max_items:
                break
    X = np.concatenate(feats, axis=0)
    Y = np.concatenate(labs, axis=0)
    if max_items is not None and X.shape[0] > max_items:
        X = X[:max_items]
        Y = Y[:max_items]
    return X, Y


def main():
    ap = argparse.ArgumentParser(description="Diagnostics: linear probe and clustering for policy_id encoding")
    ap.add_argument("--data_root", type=str, required=True)
    ap.add_argument("--run_dir", type=str, required=True)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--max_items", type=int, default=5000)
    args = ap.parse_args()

    # Normalize device aliases
    if str(args.device).lower() == "gpu":
        args.device = "cuda"

    # Build datasets
    train_index = os.path.join(args.data_root, "train", "index.json")
    test_index = os.path.join(args.data_root, "test", "index.json")
    ds_tr = NextFrameDataset(train_index)
    ds_te = NextFrameDataset(test_index)
    # Infer dims
    W = ds_tr.window
    Tteams = ds_tr[0]["state"].shape[1]
    Aagents = ds_tr[0]["state"].shape[2]
    model = GRUFeatureExtractor(teams=Tteams, agents=Aagents, hidden=128).to(args.device)

    # Features
    X_tr, y_tr = compute_features(ds_tr, model, args.device, max_items=args.max_items)
    X_te, y_te = compute_features(ds_te, model, args.device, max_items=args.max_items)

    # Linear probe
    probe, acc_tr, acc_te = train_linear_probe(X_tr, y_tr, (X_te, y_te), epochs=30, lr=0.1)

    # Clustering
    K = int(max(y_tr.max(), y_te.max())) + 1
    z_tr, _ = kmeans(X_tr, k=K, iters=50, seed=0)
    z_te, _ = kmeans(X_te, k=K, iters=50, seed=1)
    purity_tr = clustering_purity(z_tr, y_tr)
    purity_te = clustering_purity(z_te, y_te)

    report = {
        "probe_acc_train": acc_tr,
        "probe_acc_test": acc_te,
        "cluster_purity_train": purity_tr,
        "cluster_purity_test": purity_te,
        "items_train": int(X_tr.shape[0]),
        "items_test": int(X_te.shape[0]),
    }
    os.makedirs(args.run_dir, exist_ok=True)
    outp = os.path.join(args.run_dir, "diagnostics.json")
    with open(outp, "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()


