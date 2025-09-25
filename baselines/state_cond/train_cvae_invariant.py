from __future__ import annotations
import argparse
import os
import json
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from tasks.common.dataset import NextFrameDataset
from tasks.common.metrics import ade, fde
from baselines.state_cond.train_cvae import CVAE, collate, gaussian_nll, kld


class SimpleDomainEncoder(nn.Module):
    def __init__(self, teams: int, agents: int, hidden: int = 64):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(teams * agents * 2, hidden), nn.ReLU(), nn.Linear(hidden, hidden)
        )

    def forward(self, s):
        # s: [B,W,T,A,2] -> use last frame summary as domain cue
        B, W, T, A, D = s.shape
        x = s[:, -1].reshape(B, T * A * D)
        return self.fc(x)


def invariance_loss(z: torch.Tensor, d: torch.Tensor, weight: float = 1e-2) -> torch.Tensor:
    # Encourage z to be orthogonal to simple domain features d
    z = (z - z.mean(dim=0, keepdim=True))
    d = (d - d.mean(dim=0, keepdim=True))
    corr = torch.matmul(z.T, d) / max(1, z.size(0))
    return weight * (corr.pow(2).mean())


def main():
    ap = argparse.ArgumentParser(description="CVAE with invariance loss to simple domain features")
    ap.add_argument("--data_root", type=str, default="data/v1.0")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--beta", type=float, default=1e-3)
    ap.add_argument("--latent", type=int, default=16)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--inv_wt", type=float, default=1e-2)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--save_dir", type=str, default=None)
    args = ap.parse_args()

    tr = os.path.join(args.data_root, "train", "index.json")
    va = os.path.join(args.data_root, "val", "index.json")
    te = os.path.join(args.data_root, "test", "index.json")
    ds_tr, ds_va, ds_te = NextFrameDataset(tr), NextFrameDataset(va), NextFrameDataset(te)
    W = ds_tr.window
    T, A = ds_tr[0]["state"].shape[1:3]
    model = CVAE(teams=T, agents=A, window=W, hidden=args.hidden, latent=args.latent).to(args.device)
    dom_enc = SimpleDomainEncoder(T, A).to(args.device)
    opt = torch.optim.Adam(list(model.parameters()) + list(dom_enc.parameters()), lr=args.lr)
    tr_loader = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True, drop_last=True, collate_fn=collate)
    va_loader = DataLoader(ds_va, batch_size=256, collate_fn=collate)
    te_loader = DataLoader(ds_te, batch_size=256, collate_fn=collate)

    best = None
    best_state = None
    log = {"epochs": int(args.epochs), "results": []}
    for ep in range(args.epochs):
        model.train()
        dom_enc.train()
        total = 0.0
        n = 0
        for s, a in tr_loader:
            s = s.to(args.device)
            a = a.to(args.device)
            out, mu, logvar = model(s, a)
            d = dom_enc(s)
            loss = gaussian_nll(out, a) + args.beta * kld(mu, logvar) + invariance_loss(mu, d, args.inv_wt)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.item()) * s.size(0)
            n += s.size(0)
        tr_loss = total / max(1, n)
        # eval
        model.eval()
        preds, gts = [], []
        with torch.no_grad():
            for s, a in va_loader:
                s = s.to(args.device)
                mu, logvar = model.encode(s, a.to(args.device))
                out = model.decode(s, mu)
                preds.append(out[..., :2].cpu().numpy())
                gts.append(a.numpy())
        preds = np.concatenate(preds, axis=0)
        gts = np.concatenate(gts, axis=0)
        val = {"ADE": ade(preds, gts), "FDE": fde(preds, gts)}
        log["results"].append({"epoch": ep, "train": tr_loss, "val": val})
        if best is None or val["ADE"] < best["ADE"]:
            best = val
            best_state = {k: v.detach().cpu() if hasattr(v, "detach") else v for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict({k: (v.to(args.device) if hasattr(v, 'to') else v) for k, v in best_state.items()})
    # Test
    preds, gts = [], []
    with torch.no_grad():
        for s, a in te_loader:
            s = s.to(args.device)
            mu, logvar = model.encode(s, a.to(args.device))
            out = model.decode(s, mu)
            preds.append(out[..., :2].cpu().numpy())
            gts.append(a.numpy())
    preds = np.concatenate(preds, axis=0)
    gts = np.concatenate(gts, axis=0)
    test = {"ADE": ade(preds, gts), "FDE": fde(preds, gts)}
    log.update({"best_val": best, "test": test})
    print(json.dumps(log, indent=2))
    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
        with open(os.path.join(args.save_dir, "results.json"), "w") as f:
            json.dump(log, f, indent=2)
        if best_state is not None:
            torch.save(best_state, os.path.join(args.save_dir, "model_best.pt"))


if __name__ == "__main__":
    main()


