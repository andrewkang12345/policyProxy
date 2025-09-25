from __future__ import annotations
import argparse
import os
import json
import time
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from tasks.common.dataset import NextFrameDataset
from tasks.common.metrics import ade, fde


class PositionalEncoding(nn.Module):
    def __init__(self, dim: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-np.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1), :]


class TransformerActionModel(nn.Module):
    def __init__(self, teams: int, agents: int, window: int, hidden: int = 128, nhead: int = 8, layers: int = 2):
        super().__init__()
        self.teams = teams
        self.agents = agents
        self.window = window
        self.in_dim = teams * agents * 2
        self.proj = nn.Linear(self.in_dim, hidden)
        self.pos = PositionalEncoding(hidden)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden, nhead=nhead, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.head = nn.Linear(hidden, agents * 2)

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        # s: [B, W, teams, agents, 2]
        B, W, T, A, D = s.shape
        x = s.reshape(B, W, T * A * D)
        h = self.encoder(self.pos(self.proj(x)))[:, -1, :]
        out = self.head(h)
        return out.reshape(B, self.agents, 2)


def collate(batch):
    state = np.stack([b["state"] for b in batch])
    action = np.stack([b["action"] for b in batch])
    return torch.tensor(state, dtype=torch.float32), torch.tensor(action, dtype=torch.float32)


def train_epoch(model: nn.Module, loader, opt, device: str) -> float:
    model.train()
    total = 0.0
    for s, a in loader:
        s = s.to(device)
        a = a.to(device)
        pred = model(s)
        loss = ((pred - a) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        total += float(loss.item()) * s.size(0)
    return total / max(1, len(loader.dataset))


def evaluate(model: nn.Module, loader, device: str) -> dict:
    model.eval()
    preds, gts = [], []
    with torch.no_grad():
        for s, a in loader:
            s = s.to(device)
            pred = model(s)
            preds.append(pred.cpu().numpy())
            gts.append(a.numpy())
    preds = np.concatenate(preds, axis=0)
    gts = np.concatenate(gts, axis=0)
    return {"ADE": ade(preds, gts), "FDE": fde(preds, gts)}


def infer_dims(index_path: str):
    ds = NextFrameDataset(index_path)
    s0 = ds[0]["state"]
    return s0.shape[0], s0.shape[1], s0.shape[2]


def main():
    ap = argparse.ArgumentParser(description="Transformer baseline conditioned on state window (deterministic)")
    ap.add_argument("--data_root", type=str, default="data/v1.0")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--nhead", type=int, default=8)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--save_dir", type=str, default=None)
    args = ap.parse_args()

    train_index = os.path.join(args.data_root, "train", "index.json")
    val_index = os.path.join(args.data_root, "val", "index.json")
    test_index = os.path.join(args.data_root, "test", "index.json")

    ds_train = NextFrameDataset(train_index)
    ds_val = NextFrameDataset(val_index)
    ds_test = NextFrameDataset(test_index)

    W, T, A = infer_dims(train_index)
    model = TransformerActionModel(teams=T, agents=A, window=W, hidden=args.hidden, nhead=args.nhead, layers=args.layers).to(args.device)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    train_loader = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True, drop_last=True, collate_fn=collate)
    val_loader = DataLoader(ds_val, batch_size=256, collate_fn=collate)
    test_loader = DataLoader(ds_test, batch_size=256, collate_fn=collate)

    best = None
    best_state = None
    run_log = {"device": str(args.device), "epochs": int(args.epochs), "results": []}

    for ep in range(args.epochs):
        t0 = time.time()
        train_loss = train_epoch(model, train_loader, opt, args.device)
        val_metrics = evaluate(model, val_loader, args.device)
        epoch_time = time.time() - t0
        entry = {"epoch": ep, "train_mse": train_loss, "val": val_metrics, "epoch_time_sec": epoch_time}
        print(json.dumps(entry))
        run_log["results"].append(entry)
        if best is None or val_metrics["ADE"] < best["ADE"]:
            best = val_metrics
            best_state = {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in model.state_dict().items()}

    test_metrics = evaluate(model, test_loader, args.device)
    summary = {"best_val": best, "test": test_metrics}
    print(json.dumps(summary, indent=2))
    run_log.update(summary)

    if best_state is not None:
        model.load_state_dict({k: v.to(args.device) if hasattr(v, "to") else v for k, v in best_state.items()})

    ood_results = {}
    for name in os.listdir(args.data_root):
        if not name.startswith("ood_"):
            continue
        idx_path = os.path.join(args.data_root, name, "index.json")
        if not os.path.exists(idx_path):
            continue
        ds_ood = NextFrameDataset(idx_path)
        ood_loader = DataLoader(ds_ood, batch_size=256, collate_fn=collate)
        res = evaluate(model, ood_loader, args.device)
        ood_results[name] = res
    if ood_results:
        run_log["OOD"] = ood_results
        print(json.dumps({"OOD": ood_results}, indent=2))

    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
        with open(os.path.join(args.save_dir, "results.json"), "w") as f:
            json.dump(run_log, f, indent=2)
        if best_state is not None:
            torch.save(best_state, os.path.join(args.save_dir, "model_best.pt"))


if __name__ == "__main__":
    main()

