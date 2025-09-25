from __future__ import annotations
import argparse
import os
import json
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
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
    def forward(self, x):
        B, W, T, A, D = x.shape
        x = x.reshape(B, W, T * A * D)
        h, _ = self.gru(x)
        return h[:, -1, :]


def collate(batch):
    s = torch.tensor(np.stack([b["state"] for b in batch]), dtype=torch.float32)
    y = torch.tensor(np.array([b.get("intent", 0) for b in batch]), dtype=torch.long)
    return s, y


def compute_features(ds: NextFrameDataset, model: GRUFeatureExtractor, device: str, max_items: int | None = None):
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


def mean_by_label(X: np.ndarray, y: np.ndarray):
    means = {}
    for k in np.unique(y):
        means[int(k)] = X[y == k].mean(axis=0)
    return means


def cos_sim(a: np.ndarray, b: np.ndarray) -> float:
    a = a.reshape(-1)
    b = b.reshape(-1)
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def compare_splits(train_means: dict[int, np.ndarray], split_means: dict[int, np.ndarray]):
    keys = sorted(set(train_means.keys()) & set(split_means.keys()))
    if not keys:
        return {"same_mean": 0.0, "diff_mean": 0.0, "margin": 0.0}
    same = []
    diffs = []
    for k in keys:
        same.append(cos_sim(train_means[k], split_means[k]))
        # different-policy similarity: compare to all j != k and average
        others = [j for j in keys if j != k]
        if others:
            diffs.append(np.mean([cos_sim(train_means[k], split_means[j]) for j in others]))
    same_mean = float(np.mean(same)) if same else 0.0
    diff_mean = float(np.mean(diffs)) if diffs else 0.0
    return {"same_mean": same_mean, "diff_mean": diff_mean, "margin": same_mean - diff_mean}


def main():
    ap = argparse.ArgumentParser(description="Compute policy representation similarity across splits vs train")
    ap.add_argument("--data_root", type=str, required=True)
    ap.add_argument("--save_json", type=str, required=True)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--save_plot", type=str, default=None, help="Optional path (file or dir) to save margin plot")
    ap.add_argument("--max_items", type=int, default=5000)
    args = ap.parse_args()

    # Normalize device aliases
    if str(args.device).lower() == "gpu":
        args.device = "cuda"

    train_index = os.path.join(args.data_root, "train", "index.json")
    if not os.path.exists(train_index):
        raise SystemExit(f"Missing train index: {train_index}")
    base_ds = NextFrameDataset(train_index)
    W = base_ds.window
    teams = base_ds[0]["state"].shape[1]
    agents = base_ds[0]["state"].shape[2]
    model = GRUFeatureExtractor(teams=teams, agents=agents, hidden=128).to(args.device)

    X_tr, y_tr = compute_features(base_ds, model, args.device, max_items=args.max_items)
    tr_means = mean_by_label(X_tr, y_tr)

    results = {}
    # include test
    splits = ["test"] + [d for d in os.listdir(args.data_root) if d.startswith("ood_")]
    for sp in splits:
        idx = os.path.join(args.data_root, sp, "index.json")
        if not os.path.exists(idx):
            continue
        ds = NextFrameDataset(idx)
        X, y = compute_features(ds, model, args.device, max_items=args.max_items)
        sp_means = mean_by_label(X, y)
        results[sp] = compare_splits(tr_means, sp_means)

    os.makedirs(os.path.dirname(args.save_json), exist_ok=True)
    with open(args.save_json, "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))

    if args.save_plot:
        try:
            plt.switch_backend("Agg")
        except Exception:
            pass
        names = sorted(results.keys())
        margins = [results[n].get("margin", 0.0) for n in names]
        fig, ax = plt.subplots(figsize=(max(6, 1 + 0.4 * len(names)), 3.5))
        bars = ax.bar(names, margins, color=["#4c78a8" if v >= 0 else "#e45756" for v in margins])
        ax.axhline(0.0, color="k", linewidth=0.8)
        ax.set_ylabel("Rep. margin (same - diff)")
        ax.set_title("Policy representation similarity vs. train")
        ax.set_xticklabels(names, rotation=30, ha="right")
        for b, v in zip(bars, margins):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.3f}", ha="center", va="bottom" if v >= 0 else "top", fontsize=8)
        plt.tight_layout()
        # Resolve output path
        root, ext = os.path.splitext(args.save_plot)
        if ext:
            out_path = args.save_plot
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
        else:
            os.makedirs(args.save_plot, exist_ok=True)
            out_path = os.path.join(args.save_plot, "rep_similarity.png")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)


if __name__ == "__main__":
    main()


