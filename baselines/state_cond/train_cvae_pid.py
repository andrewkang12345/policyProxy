from __future__ import annotations
import argparse
import os
import json
import math
from typing import Tuple
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
import time

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from tasks.common.dataset import NextFrameDataset
from tasks.common.metrics import ade, fde


def _policy_count(dataset: NextFrameDataset) -> int:
    max_id = 0
    for ep in dataset.episodes:
        arr = ep.get("policy_ids")
        if arr is None:
            arr = ep.get("intents")
        if arr is None:
            continue
        if isinstance(arr, list):
            arr = np.array(arr)
        max_id = max(max_id, int(np.max(arr)))
    return int(max_id + 1)


class GRUEncoder(nn.Module):
    def __init__(self, teams: int, agents: int, window: int, hidden: int = 128, latent: int = 16, embed_dim: int = 16):
        super().__init__()
        self.teams = teams
        self.agents = agents
        self.window = window
        self.in_dim = teams * agents * 2
        self.embed_dim = embed_dim
        self.gru = nn.GRU(self.in_dim, hidden, batch_first=True)
        self.to_mu = nn.Linear(hidden + agents * 2 + embed_dim, latent)
        self.to_logvar = nn.Linear(hidden + agents * 2 + embed_dim, latent)

    def forward(self, s: torch.Tensor, a: torch.Tensor, emb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # s: [B,W,T,A,2], a: [B,A,2], emb: [B,embed]
        B, W, T, A, D = s.shape
        x = s.reshape(B, W, T * A * D)
        h, _ = self.gru(x)
        hs = h[:, -1, :]
        xea = torch.cat([hs, a.reshape(B, A * 2), emb], dim=-1)
        mu = self.to_mu(xea)
        logvar = self.to_logvar(xea)
        return mu, logvar


class GRUDecoder(nn.Module):
    def __init__(self, teams: int, agents: int, window: int, hidden: int = 128, latent: int = 16, embed_dim: int = 16, gaussian: bool = True):
        super().__init__()
        self.teams = teams
        self.agents = agents
        self.window = window
        self.in_dim = teams * agents * 2
        self.gaussian = gaussian
        self.embed_dim = embed_dim
        self.gru = nn.GRU(self.in_dim, hidden, batch_first=True)
        self.proj = nn.Linear(hidden + latent + embed_dim, agents * (4 if gaussian else 2))

    def forward(self, s: torch.Tensor, z: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        # s: [B,W,T,A,2], z: [B,Z], emb: [B,embed]
        B, W, T, A, D = s.shape
        x = s.reshape(B, W, T * A * D)
        h, _ = self.gru(x)
        hs = h[:, -1, :]
        y = self.proj(torch.cat([hs, z, emb], dim=-1))
        if self.gaussian:
            return y.reshape(B, self.agents, 4)
        return y.reshape(B, self.agents, 2)


class PolicyCVAE(nn.Module):
    def __init__(self, teams: int, agents: int, window: int, num_policies: int, hidden: int = 128, latent: int = 16, embed_dim: int = 16):
        super().__init__()
        self.enc = GRUEncoder(teams, agents, window, hidden, latent, embed_dim)
        self.dec = GRUDecoder(teams, agents, window, hidden, latent, embed_dim, gaussian=True)
        self.latent = latent
        self.policy_embed = nn.Embedding(num_policies, embed_dim)

    def encode(self, s: torch.Tensor, a: torch.Tensor, policy_ids: torch.Tensor):
        emb = self.policy_embed(policy_ids)
        mu, logvar = self.enc(s, a, emb)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, s: torch.Tensor, z: torch.Tensor, policy_ids: torch.Tensor):
        emb = self.policy_embed(policy_ids)
        return self.dec(s, z, emb)

    def forward(self, s: torch.Tensor, a: torch.Tensor, policy_ids: torch.Tensor):
        mu, logvar = self.encode(s, a, policy_ids)
        z = self.reparameterize(mu, logvar)
        out = self.decode(s, z, policy_ids)
        return out, mu, logvar


def collate(batch):
    state = np.stack([b["state"] for b in batch])
    action = np.stack([b["action"] for b in batch])
    policy_ids = np.array([b["policy_id"] for b in batch], dtype=np.int64)
    return (
        torch.tensor(state, dtype=torch.float32),
        torch.tensor(action, dtype=torch.float32),
        torch.tensor(policy_ids, dtype=torch.long),
    )


def gaussian_nll(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    mean = pred[..., :2]
    logvar = pred[..., 2:]
    return ((mean - target) ** 2 / (logvar.exp() + 1e-8) + logvar).mean()


def kld(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())


def evaluate(model: PolicyCVAE, loader, device: str = "cpu"):
    model.eval()
    preds = []
    gts = []
    with torch.no_grad():
        for s, a, policy_ids in loader:
            s = s.to(device)
            a = a.to(device)
            policy_ids = policy_ids.to(device)
            mu, logvar = model.encode(s, a, policy_ids)
            out = model.decode(s, mu, policy_ids)
            mean = out[..., :2].cpu().numpy()
            preds.append(mean)
            gts.append(a.cpu().numpy())
    preds = np.concatenate(preds, axis=0)
    gts = np.concatenate(gts, axis=0)
    return {"ADE": ade(preds, gts), "FDE": fde(preds, gts)}


def main():
    ap = argparse.ArgumentParser(description="Policy-aware state-conditional CVAE")
    ap.add_argument("--data_root", type=str, default="data/v1.0")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--beta", type=float, default=1e-3, help="KL weight")
    ap.add_argument("--latent", type=int, default=16)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--policy_embed_dim", type=int, default=16)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--save_dir", type=str, default=None)
    args = ap.parse_args()

    train_index = os.path.join(args.data_root, "train", "index.json")
    val_index = os.path.join(args.data_root, "val", "index.json")
    test_index = os.path.join(args.data_root, "test", "index.json")

    ds_train = NextFrameDataset(train_index)
    ds_val = NextFrameDataset(val_index)
    ds_test = NextFrameDataset(test_index)

    W = ds_train.window
    teams = ds_train[0]["state"].shape[1]
    agents = ds_train[0]["state"].shape[2]
    num_policies = _policy_count(ds_train)

    model = PolicyCVAE(
        teams=teams,
        agents=agents,
        window=W,
        num_policies=num_policies,
        hidden=args.hidden,
        latent=args.latent,
        embed_dim=args.policy_embed_dim,
    ).to(args.device)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    train_loader = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True, drop_last=True, collate_fn=collate)
    val_loader = DataLoader(ds_val, batch_size=256, collate_fn=collate)
    test_loader = DataLoader(ds_test, batch_size=256, collate_fn=collate)

    best = None
    best_state = None
    log = {
        "epochs": int(args.epochs),
        "results": [],
        "num_policies": num_policies,
        "policy_embed_dim": int(args.policy_embed_dim),
        "latent": int(args.latent),
        "hidden": int(args.hidden),
    }

    for ep in range(args.epochs):
        t0 = time.time()
        model.train()
        total = 0.0
        n = 0
        for s, a, policy_ids in train_loader:
            s = s.to(args.device)
            a = a.to(args.device)
            policy_ids = policy_ids.to(args.device)
            out, mu, logvar = model(s, a, policy_ids)
            loss_rec = gaussian_nll(out, a)
            loss_kl = kld(mu, logvar)
            loss = loss_rec + args.beta * loss_kl
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.item()) * s.size(0)
            n += s.size(0)
        tr_loss = total / max(1, n)
        val = evaluate(model, val_loader, device=args.device)
        epoch_time = time.time() - t0
        payload = {
            "epoch": ep,
            "train_elbo": tr_loss,
            "val": val,
            "epoch_time_sec": epoch_time,
        }
        print(
            f"Epoch {ep}/{args.epochs-1}: train_elbo={tr_loss:.4f}, val_ADE={val['ADE']:.4f}, val_FDE={val['FDE']:.4f}, time={epoch_time:.1f}s"
        )
        log["results"].append(payload)
        if best is None or val["ADE"] < best["ADE"]:
            best = val
            best_state = {k: v.detach().cpu() if hasattr(v, "detach") else v for k, v in model.state_dict().items()}

    if best is None:
        best = {"ADE": math.nan, "FDE": math.nan}

    print(f"Training completed. Best val ADE: {best['ADE']:.4f}")

    if best_state is not None:
        model.load_state_dict({k: (v.to(args.device) if hasattr(v, "to") else v) for k, v in best_state.items()})
    test = evaluate(model, test_loader, device=args.device)
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
