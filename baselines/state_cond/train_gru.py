from __future__ import annotations
import argparse
import os
import json
import time
import numpy as np
from tqdm import tqdm
import torch
from torch import nn
from torch.utils.data import DataLoader
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from tasks.common.dataset import NextFrameDataset
from tasks.common.metrics import ade, fde


class GRUModel(nn.Module):
    def __init__(self, teams: int, agents: int, hidden: int = 128, gaussian_head: bool = False):
        super().__init__()
        # Input per frame: teams*agents*2
        self.teams = teams
        self.agents = agents
        self.in_dim = teams * agents * 2
        self.gru = nn.GRU(self.in_dim, hidden, batch_first=True)
        self.gaussian = gaussian_head
        if self.gaussian:
            self.head = nn.Linear(hidden, agents * 4)  # mean(2) + logvar(2) per agent
        else:
            self.head = nn.Linear(hidden, agents * 2)

    def forward(self, x):
        # x: [B, W, teams, agents, 2]
        B, W, T, A, D = x.shape
        x = x.reshape(B, W, T * A * D)
        h, _ = self.gru(x)
        y = self.head(h[:, -1, :])
        if self.gaussian:
            y = y.reshape(B, self.agents, 4)
        else:
            y = y.reshape(B, self.agents, 2)
        return y


def collate(batch):
    state = np.stack([b["state"] for b in batch])
    action = np.stack([b["action"] for b in batch])
    return torch.tensor(state, dtype=torch.float32), torch.tensor(action, dtype=torch.float32)


def train_epoch(model, loader, opt, device="cpu"):
    model.train()
    total = 0.0
    for s, a in loader:
        s = s.to(device)
        a = a.to(device)
        pred = model(s)
        if getattr(model, 'gaussian', False):
            mean = pred[..., :2]
            logvar = pred[..., 2:]
            loss = ((mean - a) ** 2 / (logvar.exp() + 1e-8) + logvar).mean()
        else:
            loss = ((pred - a) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        total += loss.item() * s.size(0)
    return total / len(loader.dataset)


def evaluate(model, loader, device="cpu"):
    model.eval()
    preds = []
    gts = []
    with torch.no_grad():
        for s, a in loader:
            s = s.to(device)
            pred = model(s)
            if getattr(model, 'gaussian', False):
                mean = pred[..., :2].cpu().numpy()
                preds.append(mean)
            else:
                preds.append(pred.cpu().numpy())
            gts.append(a.numpy())
    preds = np.concatenate(preds, axis=0)
    gts = np.concatenate(gts, axis=0)
    out = {"ADE": ade(preds, gts), "FDE": fde(preds, gts)}
    return out


def infer_dims(index_path: str):
    ds = NextFrameDataset(index_path)
    s0 = ds[0]["state"]  # [W, teams, agents, 2]
    return s0.shape[0], s0.shape[1], s0.shape[2]  # W, teams, agents


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, default="data/v1.0")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--save_dir", type=str, default=None)
    ap.add_argument("--gaussian_head", action="store_true")
    args = ap.parse_args()

    train_index = os.path.join(args.data_root, "train", "index.json")
    val_index = os.path.join(args.data_root, "val", "index.json")
    test_index = os.path.join(args.data_root, "test", "index.json")

    ds_train = NextFrameDataset(train_index)
    ds_val = NextFrameDataset(val_index)
    ds_test = NextFrameDataset(test_index)

    W, T, A = infer_dims(train_index)
    model = GRUModel(teams=T, agents=A, gaussian_head=args.gaussian_head)
    model.to(args.device)
    total_params = count_parameters(model)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    train_loader = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True, drop_last=True, collate_fn=collate)
    val_loader = DataLoader(ds_val, batch_size=256, collate_fn=collate)
    test_loader = DataLoader(ds_test, batch_size=256, collate_fn=collate)

    best = None
    best_state = None
    run_log = {
        "device": str(args.device),
        "params": int(total_params),
        "epochs": int(args.epochs),
        "results": []
    }
    for ep in range(args.epochs):
        t0 = time.time()
        loss = train_epoch(model, train_loader, opt, device=args.device)
        val = evaluate(model, val_loader, device=args.device)
        epoch_time = time.time() - t0
        payload = {"epoch": ep, "train_mse": loss, "val": val, "epoch_time_sec": epoch_time}
        print(json.dumps(payload))
        run_log["results"].append(payload)
        if best is None or val["ADE"] < best["ADE"]:
            best = val
            best_state = {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in model.state_dict().items()}

    test = evaluate(model, test_loader, device=args.device)
    summary = {"best_val": best, "test": test}
    print(json.dumps(summary, indent=2))
    run_log.update(summary)

    # Evaluate on available OOD splits under data_root
    ood_results = {}
    for name in os.listdir(args.data_root):
        if not name.startswith("ood_"):
            continue
        idx = os.path.join(args.data_root, name, "index.json")
        if not os.path.exists(idx):
            continue
        ds_ood = NextFrameDataset(idx)
        ood_loader = DataLoader(ds_ood, batch_size=256, collate_fn=collate)
        if best_state is not None:
            model.load_state_dict({k: v.to(args.device) if hasattr(v, 'to') else v for k, v in best_state.items()})
        res = evaluate(model, ood_loader, device=args.device)
        ood_results[name] = res
    if ood_results:
        print(json.dumps({"OOD": ood_results}, indent=2))
        run_log["OOD"] = ood_results

    # Save artifacts
    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
        with open(os.path.join(args.save_dir, "results.json"), "w") as f:
            json.dump(run_log, f, indent=2)
        if best_state is not None:
            torch.save(best_state, os.path.join(args.save_dir, "model_best.pt"))


if __name__ == "__main__":
    main()
