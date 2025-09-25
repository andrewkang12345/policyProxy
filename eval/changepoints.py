from __future__ import annotations
import argparse
import json
import os
from typing import List, Dict, Any

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

    def forward(self, x):
        B, W, T, A, D = x.shape
        x = x.reshape(B, W, T * A * D)
        h, _ = self.gru(x)
        return h


def collate(batch):
    s = np.stack([b["state"] for b in batch])
    y = [b["policy_id"] for b in batch]
    return torch.tensor(s, dtype=torch.float32), y


def energy_score(features: torch.Tensor, win: int = 10) -> torch.Tensor:
    T = features.size(0)
    scores = torch.zeros(T, device=features.device)
    for t in range(win, T - win):
        past = features[t - win : t]
        future = features[t : t + win]
        diff = past.mean(dim=0) - future.mean(dim=0)
        scores[t] = torch.norm(diff, p=2)
    return scores


def detect_changepoints(scores: torch.Tensor, tau: float = 3.0, cooldown: int = 5) -> List[int]:
    cps = []
    last = -cooldown
    thresh = scores.mean() + tau * scores.std()
    for t, s in enumerate(scores):
        if t - last < cooldown:
            continue
        if s >= thresh:
            cps.append(t)
            last = t
    return cps


def changepoint_metrics(pred: List[int], truth: List[int], tolerance: int = 3) -> Dict[str, Any]:
    pred = sorted(pred)
    truth = sorted(truth)
    if not truth:
        return {"f1_tau3": 0.0, "mabe": None, "delay_mean": None, "count_pred": len(pred)}
    tp = 0
    delays = []
    errors = []
    for cp in truth:
        best = None
        for p in pred:
            if abs(p - cp) <= tolerance:
                best = p
                break
        if best is not None:
            tp += 1
            delays.append(max(0, best - cp))
            errors.append(abs(best - cp))
    precision = tp / max(1, len(pred))
    recall = tp / max(1, len(truth))
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    mabe = float(np.mean(errors)) if errors else None
    delay_mean = float(np.mean(delays)) if delays else None
    return {
        "f1_tau3": f1,
        "mabe": mabe,
        "delay_mean": delay_mean,
        "count_pred": len(pred),
        "count_truth": len(truth),
    }


def compute_changepoints(dataset: NextFrameDataset, model: GRUFeatureExtractor, device: str, tau: float, cooldown: int) -> Dict[str, Any]:
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate)
    metrics_list = []
    model.eval()
    with torch.no_grad():
        for s, y in loader:
            s = s.to(device)
            feats = model(s)[0]
            scores = energy_score(feats)
            pred_cps = detect_changepoints(scores, tau=tau, cooldown=cooldown)
            policy_seq = y[0]
            truth_cps = []
            prev = int(policy_seq[0]) if len(policy_seq) > 0 else 0
            for t in range(1, len(policy_seq)):
                cur = int(policy_seq[t])
                if cur != prev:
                    truth_cps.append(t)
                    prev = cur
            metrics_list.append(changepoint_metrics(pred_cps, truth_cps))
    agg = {}
    for m in metrics_list[0].keys():
        values = [entry[m] for entry in metrics_list if entry[m] is not None]
        if values:
            agg[m] = float(np.mean(values))
    return {"per_episode": metrics_list, "aggregate": agg}


def main():
    ap = argparse.ArgumentParser(description="Changepoint detection evaluation")
    ap.add_argument("--data_root", type=str, required=True)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--tau", type=float, default=3.0)
    ap.add_argument("--cooldown", type=int, default=5)
    ap.add_argument("--save_json", type=str, required=True)
    args = ap.parse_args()

    if str(args.device).lower() == "gpu":
        args.device = "cuda"

    index_path = os.path.join(args.data_root, "index.json")
    if not os.path.exists(index_path):
        raise FileNotFoundError(index_path)
    dataset = NextFrameDataset(index_path, label_kind="policy")
    W = dataset.window
    teams = dataset[0]["state"].shape[1]
    agents = dataset[0]["state"].shape[2]
    model = GRUFeatureExtractor(teams=teams, agents=agents, hidden=128).to(args.device)

    metrics = compute_changepoints(dataset, model, args.device, tau=args.tau, cooldown=args.cooldown)
    os.makedirs(os.path.dirname(args.save_json), exist_ok=True)
    with open(args.save_json, "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
