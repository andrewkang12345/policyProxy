from __future__ import annotations
import argparse
import os
import json
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from tasks.common.dataset import NextFrameDataset
from baselines.state_cond.train_cvae import CVAE, collate, evaluate


class LinearProbe(nn.Module):
    def __init__(self, in_dim: int, num_classes: int):
        super().__init__()
        self.lin = nn.Linear(in_dim, num_classes)
    def forward(self, x):
        return self.lin(x)


def build_features(model: CVAE, loader, device: str) -> tuple[np.ndarray, np.ndarray, int]:
    model.eval()
    feats = []
    labels = []
    Aagents = None
    with torch.no_grad():
        for batch in loader:
            s, a, y = batch
            s = s.to(device)
            a = a.to(device)
            # deterministic decode using mu
            mu, _ = model.encode(s, a)
            out = model.decode(s, mu)
            mean = out[..., :2]
            if Aagents is None:
                Aagents = mean.size(1)
            x = mean.reshape(mean.size(0), -1).cpu().numpy()
            feats.append(x)
            labels.append(y.numpy())
    X = np.concatenate(feats, axis=0)
    y = np.concatenate(labels, axis=0)
    num_classes = int(y.max()) + 1
    return X, y, num_classes


def build_features_segmented(model: CVAE, index_path: str, device: str) -> tuple[np.ndarray, np.ndarray, int]:
    # Build features by aggregating over contiguous segments where policy_ids are constant
    from tasks.common.dataset import load_index, load_episode
    model.eval()
    Xs = []
    Ys = []
    with torch.no_grad():
        index = load_index(index_path)
        for item in index:
            ep = load_episode(item["path"])  # has pos [T,teams,agents,2], vel [T,teams,agents,2], intents [T]
            pos = ep["pos"]
            vel = ep["vel"]
            intents = ep["intents"].astype(int)
            W = int(ep["meta"].get("window", 6))
            T = pos.shape[0]
            # find segments
            start = W
            while start < T - 1:
                seg_label = intents[start]
                end = start + 1
                while end < T - 1 and intents[end] == seg_label:
                    end += 1
                # collect windows within [start, end)
                feats = []
                for t in range(start, end):
                    s = torch.tensor(pos[t - W : t][None, ...], dtype=torch.float32, device=device)
                    a = torch.tensor(vel[t, 0][None, ...], dtype=torch.float32, device=device)
                    mu, _ = model.encode(s, a)
                    out = model.decode(s, mu)
                    mean = out[..., :2].reshape(1, -1).cpu().numpy()
                    feats.append(mean)
                if len(feats) > 0:
                    seg_feat = np.mean(np.concatenate(feats, axis=0), axis=0)
                    Xs.append(seg_feat)
                    Ys.append(int(seg_label))
                start = end
    X = np.stack(Xs) if len(Xs) > 0 else np.zeros((0, model.dec.agents * 2))
    y = np.array(Ys, dtype=int) if len(Ys) > 0 else np.zeros((0,), dtype=int)
    K = int(y.max()) + 1 if y.size > 0 else 0
    return X, y, K


def train_probe(X: np.ndarray, y: np.ndarray, num_classes: int, epochs: int = 50, lr: float = 1e-2, device: str = "cpu") -> LinearProbe:
    X_t = torch.tensor(X, dtype=torch.float32, device=device)
    y_t = torch.tensor(y, dtype=torch.long, device=device)
    probe = LinearProbe(X.shape[1], num_classes).to(device)
    opt = torch.optim.Adam(probe.parameters(), lr=lr)
    for _ in range(epochs):
        opt.zero_grad()
        logits = probe(X_t)
        loss = nn.CrossEntropyLoss()(logits, y_t)
        loss.backward()
        opt.step()
    return probe


def eval_probe(probe: LinearProbe, X: np.ndarray, y: np.ndarray, device: str = "cpu") -> float:
    with torch.no_grad():
        logits = probe(torch.tensor(X, dtype=torch.float32, device=device))
        pred = logits.argmax(dim=-1).cpu().numpy()
    return float((pred == y).mean())


def main():
    ap = argparse.ArgumentParser(description="Evaluate CVAE policy representation via linear probe on predicted actions")
    ap.add_argument("--data_root", type=str, required=True)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--save_json", type=str, default=None)
    ap.add_argument("--mode", type=str, default="unknown_segments", choices=["unknown_segments", "known_segments"], help="Policy repr: per-step (unknown) vs per-segment (known)")
    args = ap.parse_args()

    device = args.device
    if str(device).lower() == "gpu":
        device = "cuda"

    train_index = os.path.join(args.data_root, "train", "index.json")
    ds_train = NextFrameDataset(train_index)
    W = ds_train.window
    Tteams = ds_train[0]["state"].shape[1]
    Aagents = ds_train[0]["state"].shape[2]

    model = CVAE(teams=Tteams, agents=Aagents, window=W, hidden=128, latent=16, global_z=True).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    train_loader_small = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True, drop_last=True, collate_fn=collate)
    # quick warmup
    for _ in range(args.epochs):
        for s, a in train_loader_small:
            s = s.to(device)
            a = a.to(device)
            out, mu, logvar = model(s, a)
            from baselines.state_cond.train_cvae import gaussian_nll, kld
            loss = gaussian_nll(out, a) + 1e-3 * kld(mu, logvar)
            opt.zero_grad()
            loss.backward()
            opt.step()

    # Build features
    # Collate that preserves intent labels (per-step mode)
    def collate_sa_y(batch):
        import numpy as _np
        states = _np.stack([b["state"] for b in batch])
        actions = _np.stack([b["action"] for b in batch])
        intents = _np.stack([b["intent"] for b in batch])
        return (
            torch.tensor(states, dtype=torch.float32),
            torch.tensor(actions, dtype=torch.float32),
            torch.tensor(intents, dtype=torch.long),
        )

    if args.mode == "known_segments":
        Xtr, ytr, K = build_features_segmented(model, train_index, device=device)
    else:
        train_loader = DataLoader(ds_train, batch_size=args.batch_size, collate_fn=collate_sa_y)
        Xtr, ytr, K = build_features(model, train_loader, device=device)
    probe = train_probe(Xtr, ytr, num_classes=K, epochs=100, lr=5e-3, device=device)

    results = {}
    for split in sorted(os.listdir(args.data_root)):
        sp = os.path.join(args.data_root, split)
        idx = os.path.join(sp, "index.json")
        if not os.path.isdir(sp) or not os.path.exists(idx):
            continue
        if args.mode == "known_segments":
            X, y, _ = build_features_segmented(model, idx, device=device)
        else:
            ds = NextFrameDataset(idx)
            loader = DataLoader(ds, batch_size=args.batch_size, collate_fn=collate_sa_y)
            X, y, _ = build_features(model, loader, device=device)
        acc = eval_probe(probe, X, y, device=device)
        results[split] = {"probe_accuracy": acc}

    print(json.dumps(results, indent=2))
    if args.save_json:
        os.makedirs(os.path.dirname(args.save_json), exist_ok=True)
        with open(args.save_json, "w") as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()


