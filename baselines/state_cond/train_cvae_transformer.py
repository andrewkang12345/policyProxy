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


class PositionalEncoding(nn.Module):
    def __init__(self, dim: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-np.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0), persistent=False)

    def forward(self, x):
        # x: [B, W, D]
        W = x.size(1)
        return x + self.pe[:, :W, :]


class TransEnc(nn.Module):
    def __init__(self, in_dim: int, agents: int, latent: int = 16, nhead: int = 8, num_layers: int = 2, hidden: int = 128):
        super().__init__()
        self.proj = nn.Linear(in_dim, hidden)
        self.pos = PositionalEncoding(hidden)
        enc_layer = nn.TransformerEncoderLayer(d_model=hidden, nhead=nhead, batch_first=True)
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.to_mu = nn.Linear(hidden + agents * 2, latent)
        self.to_logvar = nn.Linear(hidden + agents * 2, latent)

    def forward(self, s, a):
        # s: [B,W,T,A,2], a: [B,A,2]
        B, W, T, A, D = s.shape
        x = s.reshape(B, W, T * A * D)
        x = self.pos(self.proj(x))
        h = self.enc(x)[:, -1, :]
        xa = torch.cat([h, a.reshape(B, A * 2)], dim=-1)
        mu = self.to_mu(xa)
        logvar = self.to_logvar(xa)
        return mu, logvar


class TransDec(nn.Module):
    def __init__(self, in_dim: int, agents: int, latent: int = 16, nhead: int = 8, num_layers: int = 2, hidden: int = 128, gaussian: bool = True):
        super().__init__()
        self.gaussian = gaussian
        self.proj = nn.Linear(in_dim, hidden)
        self.pos = PositionalEncoding(hidden)
        enc_layer = nn.TransformerEncoderLayer(d_model=hidden, nhead=nhead, batch_first=True)
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.out = nn.Linear(hidden + latent, agents * (4 if gaussian else 2))

    def forward(self, s, z):
        B, W, T, A, D = s.shape
        x = s.reshape(B, W, T * A * D)
        h = self.enc(self.pos(self.proj(x)))[:, -1, :]
        y = self.out(torch.cat([h, z], dim=-1))
        if self.gaussian:
            return y.reshape(B, A, 4)
        return y.reshape(B, A, 2)


class TransCVAE(nn.Module):
    def __init__(self, teams: int, agents: int, window: int, hidden: int = 128, latent: int = 16, nhead: int = 8, layers: int = 2):
        super().__init__()
        in_dim = teams * agents * 2
        self.enc = TransEnc(in_dim, agents, latent, nhead, layers, hidden)
        self.dec = TransDec(in_dim, agents, latent, nhead, layers, hidden, gaussian=True)

    def encode(self, s, a):
        return self.enc(s, a)
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std
    def decode(self, s, z):
        return self.dec(s, z)
    def forward(self, s, a):
        mu, logvar = self.encode(s, a)
        z = self.reparameterize(mu, logvar)
        out = self.decode(s, z)
        return out, mu, logvar


def collate(batch):
    s = np.stack([b["state"] for b in batch])
    a = np.stack([b["action"] for b in batch])
    return torch.tensor(s, dtype=torch.float32), torch.tensor(a, dtype=torch.float32)


def gaussian_nll(pred, target):
    mean = pred[..., :2]
    logvar = pred[..., 2:]
    return ((mean - target) ** 2 / (logvar.exp() + 1e-8) + logvar).mean()


def kld(mu, logvar):
    return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())


def evaluate(model, loader, device="cpu"):
    model.eval()
    preds, gts = [], []
    with torch.no_grad():
        for s, a in loader:
            s = s.to(device)
            mu, logvar = model.encode(s, a.to(device))
            out = model.decode(s, mu)
            preds.append(out[..., :2].cpu().numpy())
            gts.append(a.numpy())
    preds = np.concatenate(preds, axis=0)
    gts = np.concatenate(gts, axis=0)
    return {"ADE": ade(preds, gts), "FDE": fde(preds, gts)}


def main():
    ap = argparse.ArgumentParser(description="Transformer CVAE (state-conditional)")
    ap.add_argument("--data_root", type=str, default="data/v1.0")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--beta", type=float, default=1e-3)
    ap.add_argument("--latent", type=int, default=16)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--nhead", type=int, default=8)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--save_dir", type=str, default=None)
    args = ap.parse_args()

    tr = os.path.join(args.data_root, "train", "index.json")
    va = os.path.join(args.data_root, "val", "index.json")
    te = os.path.join(args.data_root, "test", "index.json")
    ds_tr, ds_va, ds_te = NextFrameDataset(tr), NextFrameDataset(va), NextFrameDataset(te)
    W = ds_tr.window
    T, A = ds_tr[0]["state"].shape[1:3]
    model = TransCVAE(teams=T, agents=A, window=W, hidden=args.hidden, latent=args.latent, nhead=args.nhead, layers=args.layers).to(args.device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    tr_loader = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True, drop_last=True, collate_fn=collate)
    va_loader = DataLoader(ds_va, batch_size=256, collate_fn=collate)
    te_loader = DataLoader(ds_te, batch_size=256, collate_fn=collate)

    best = None
    best_state = None
    log = {"epochs": int(args.epochs), "results": []}
    for ep in range(args.epochs):
        model.train()
        total = 0.0
        n = 0
        for s, a in tr_loader:
            s = s.to(args.device)
            a = a.to(args.device)
            out, mu, logvar = model(s, a)
            loss = gaussian_nll(out, a) + args.beta * kld(mu, logvar)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.item()) * s.size(0)
            n += s.size(0)
        tr_loss = total / max(1, n)
        val = evaluate(model, va_loader, device=args.device)
        log["results"].append({"epoch": ep, "train_elbo": tr_loss, "val": val})
        if best is None or val["ADE"] < best["ADE"]:
            best = val
            best_state = {k: v.detach().cpu() if hasattr(v, "detach") else v for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict({k: (v.to(args.device) if hasattr(v, 'to') else v) for k, v in best_state.items()})
    test = evaluate(model, te_loader, device=args.device)
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


