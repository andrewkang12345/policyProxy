#!/usr/bin/env python3
"""
Fixed-Z CVAE for Policy-or-Proxy Benchmarking

This implementation uses a single fixed latent z across segments with known fixed policies,
as specified for the robustness benchmarking experiments.
"""

from __future__ import annotations
import argparse
import os
import json
import time
from typing import Tuple, Dict, Any
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from tasks.common.dataset import NextFrameDataset
from tasks.common.metrics import ade, fde


def _policy_count(dataset: NextFrameDataset) -> int:
    """Count unique policy IDs in dataset."""
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
    def __init__(self, teams: int, agents: int, window: int, hidden: int = 128, latent: int = 16):
        super().__init__()
        self.teams = teams
        self.agents = agents
        self.window = window
        self.in_dim = teams * agents * 2
        self.gru = nn.GRU(self.in_dim, hidden, batch_first=True)
        self.to_mu = nn.Linear(hidden + agents * 2, latent)
        self.to_logvar = nn.Linear(hidden + agents * 2, latent)

    def forward(self, s: torch.Tensor, a: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # s: [B,W,T,A,2], a: [B,A,2]
        B, W, T, A, D = s.shape
        x = s.reshape(B, W, T * A * D)
        h, _ = self.gru(x)
        hs = h[:, -1, :]  # [B,H]
        xea = torch.cat([hs, a.reshape(B, A * 2)], dim=-1)
        mu = self.to_mu(xea)
        logvar = self.to_logvar(xea)
        return mu, logvar


class GRUDecoder(nn.Module):
    def __init__(self, teams: int, agents: int, window: int, hidden: int = 128, latent: int = 16):
        super().__init__()
        self.teams = teams
        self.agents = agents
        self.window = window
        self.in_dim = teams * agents * 2
        self.gru = nn.GRU(self.in_dim, hidden, batch_first=True)
        self.proj = nn.Linear(hidden + latent, agents * 4)  # mean(2) + logvar(2)

    def forward(self, s: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        # s: [B,W,T,A,2], z: [B,Z]
        B, W, T, A, D = s.shape
        x = s.reshape(B, W, T * A * D)
        h, _ = self.gru(x)
        hs = h[:, -1, :]
        y = self.proj(torch.cat([hs, z], dim=-1))
        return y.reshape(B, self.agents, 4)  # mean(2) + logvar(2)


class FixedZCVAE(nn.Module):
    """
    CVAE that uses fixed latent z per policy segment.
    
    Key features:
    - Learns a fixed z for each policy ID
    - Same z used across all samples from same policy
    - Supports both action rollout and representation tasks
    """
    
    def __init__(self, teams: int, agents: int, window: int, num_policies: int, 
                 hidden: int = 128, latent: int = 16, variant: str = "baseline"):
        super().__init__()
        self.enc = GRUEncoder(teams, agents, window, hidden, latent)
        self.dec = GRUDecoder(teams, agents, window, hidden, latent)
        self.latent = latent
        self.num_policies = num_policies
        self.variant = variant
        
        # Fixed latent vectors per policy ID
        self.policy_z = nn.Parameter(torch.randn(num_policies, latent) * 0.1)
        
        # Variant-specific components
        if variant == "policy_conditional":
            # Policy ID embedding for additional conditioning
            self.policy_embed = nn.Embedding(num_policies, 16)
            self.policy_proj = nn.Linear(16, latent)
        elif variant == "learned_repr":
            # Learnable policy representations
            self.repr_net = nn.Sequential(
                nn.Linear(latent, hidden),
                nn.ReLU(),
                nn.Linear(hidden, latent)
            )

    def get_policy_z(self, policy_ids: torch.Tensor) -> torch.Tensor:
        """Get fixed z for given policy IDs."""
        # policy_ids: [B] -> z: [B, latent]
        z = self.policy_z[policy_ids]
        
        if self.variant == "policy_conditional":
            # Add policy embedding information
            emb = self.policy_embed(policy_ids)
            z_cond = self.policy_proj(emb)
            z = z + z_cond
        elif self.variant == "learned_repr":
            # Apply representation learning
            z = self.repr_net(z)
            
        return z

    def encode(self, s: torch.Tensor, a: torch.Tensor, policy_ids: torch.Tensor):
        """Encode state-action to latent (but return fixed z instead)."""
        # Traditional encoding (for regularization)
        mu, logvar = self.enc(s, a)
        
        # Get fixed z for policy IDs
        z_fixed = self.get_policy_z(policy_ids)
        
        return z_fixed, mu, logvar

    def decode(self, s: torch.Tensor, z: torch.Tensor):
        """Decode from latent to action."""
        return self.dec(s, z)

    def forward(self, s: torch.Tensor, a: torch.Tensor, policy_ids: torch.Tensor, 
                task: str = "action_rollout"):
        """
        Forward pass for different tasks.
        
        Args:
            task: "action_rollout" or "representation"
        """
        z_fixed, mu_enc, logvar_enc = self.encode(s, a, policy_ids)
        
        if task == "action_rollout":
            # Use fixed z for action prediction
            out = self.decode(s, z_fixed)
            return out, z_fixed, mu_enc, logvar_enc
        elif task == "representation":
            # Return representations for downstream tasks
            return z_fixed, mu_enc, logvar_enc
        else:
            raise ValueError(f"Unknown task: {task}")


def collate_with_policy_ids(batch):
    """Collate function that includes policy IDs."""
    states = torch.stack([torch.tensor(b["state"], dtype=torch.float32) for b in batch])
    actions = torch.stack([torch.tensor(b["action"], dtype=torch.float32) for b in batch])
    
    # Extract policy IDs
    policy_ids = []
    for b in batch:
        pid = b.get("policy_id")
        if pid is None:
            pid = b.get("intent", 0)  # fallback
        policy_ids.append(int(pid))
    policy_ids = torch.tensor(policy_ids, dtype=torch.long)
    
    return states, actions, policy_ids


def compute_elbo_loss(recon_out: torch.Tensor, target_actions: torch.Tensor,
                     z_fixed: torch.Tensor, mu_enc: torch.Tensor, logvar_enc: torch.Tensor,
                     beta: float = 1.0) -> torch.Tensor:
    """Compute ELBO loss for fixed-z CVAE."""
    
    # Reconstruction loss (negative log-likelihood)
    pred_mean = recon_out[:, :, :2]  # [B, A, 2]
    pred_logvar = recon_out[:, :, 2:]  # [B, A, 2]
    
    diff = target_actions - pred_mean
    var = torch.exp(pred_logvar)
    nll = 0.5 * (pred_logvar + diff**2 / var + np.log(2 * np.pi))
    recon_loss = nll.sum()
    
    # KL divergence (regularize encoded mu/logvar toward fixed z)
    kl_loss = -0.5 * torch.sum(1 + logvar_enc - mu_enc**2 - torch.exp(logvar_enc))
    
    # Regularization: keep fixed z close to encoded representations
    z_reg_loss = torch.nn.functional.mse_loss(z_fixed, mu_enc)
    
    elbo = recon_loss + beta * kl_loss + 0.1 * z_reg_loss
    return elbo


def train_epoch(model: FixedZCVAE, loader: DataLoader, optimizer: torch.optim.Optimizer,
                beta: float = 1.0, device: str = "cpu") -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    count = 0
    
    for states, actions, policy_ids in loader:
        states = states.to(device)
        actions = actions.to(device)
        policy_ids = policy_ids.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        recon_out, z_fixed, mu_enc, logvar_enc = model(states, actions, policy_ids, task="action_rollout")
        
        # Compute loss
        loss = compute_elbo_loss(recon_out, actions, z_fixed, mu_enc, logvar_enc, beta)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        count += 1
    
    return total_loss / count if count > 0 else 0.0


def evaluate(model: FixedZCVAE, loader: DataLoader, device: str = "cpu") -> Dict[str, float]:
    """Evaluate model on validation/test set."""
    model.eval()
    total_ade = 0.0
    total_fde = 0.0
    count = 0
    
    with torch.no_grad():
        for states, actions, policy_ids in loader:
            states = states.to(device)
            actions = actions.to(device) 
            policy_ids = policy_ids.to(device)
            
            # Forward pass
            recon_out, _, _, _ = model(states, actions, policy_ids, task="action_rollout")
            pred_actions = recon_out[:, :, :2]  # Take mean predictions
            
            # Compute metrics
            batch_ade = ade(pred_actions, actions)
            batch_fde = fde(pred_actions, actions)
            
            # Handle both tensor and float returns
            total_ade += batch_ade.item() if hasattr(batch_ade, 'item') else batch_ade
            total_fde += batch_fde.item() if hasattr(batch_fde, 'item') else batch_fde
            count += 1
    
    return {
        "ADE": total_ade / count if count > 0 else float('inf'),
        "FDE": total_fde / count if count > 0 else float('inf')
    }


def main():
    parser = argparse.ArgumentParser(description="Train Fixed-Z CVAE")
    parser.add_argument("--data_root", type=str, required=True, help="Path to dataset root")
    parser.add_argument("--save_dir", type=str, required=True, help="Directory to save model")
    parser.add_argument("--variant", type=str, default="baseline", 
                       choices=["baseline", "policy_conditional", "learned_repr"],
                       help="CVAE variant")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--beta", type=float, default=1e-3, help="KL weight")
    parser.add_argument("--latent", type=int, default=16, help="Latent dimension")
    parser.add_argument("--hidden", type=int, default=128, help="Hidden dimension")
    parser.add_argument("--device", type=str, default="cpu", help="Device")
    
    # New arguments for proper task framework
    parser.add_argument("--task", type=str, default="action_rollout", choices=["action_rollout", "representation"],
                       help="Task type")
    parser.add_argument("--use_gt_policy_as_z", action="store_true", help="Use GT policy ID as latent z (Task A1)")
    parser.add_argument("--segment_aware", action="store_true", help="Use segment-aware evaluation")
    parser.add_argument("--pretrained_repr_path", type=str, help="Path to pretrained policy representation model (Task A2)")
    parser.add_argument("--policy_embed_dim", type=int, default=8, help="Policy embedding dimension")
    
    args = parser.parse_args()
    
    # Device setup
    device = torch.device(args.device)
    
    # Load datasets (NextFrameDataset expects index.json file path, not directory)
    train_index = os.path.join(args.data_root, "train", "index.json")
    val_index = os.path.join(args.data_root, "val", "index.json") 
    test_index = os.path.join(args.data_root, "test", "index.json")
    
    # Check if the index files exist
    if not os.path.exists(train_index):
        raise FileNotFoundError(f"No index.json found at {train_index}. Make sure data generation completed successfully.")
        
    train_dataset = NextFrameDataset(train_index)
    val_dataset = NextFrameDataset(val_index) 
    test_dataset = NextFrameDataset(test_index)
    
    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, 
                             collate_fn=collate_with_policy_ids)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                           collate_fn=collate_with_policy_ids)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                            collate_fn=collate_with_policy_ids)
    
    # Model setup
    num_policies = _policy_count(train_dataset)
    teams, agents, window = 2, 3, 6  # From config
    
    model = FixedZCVAE(
        teams=teams, agents=agents, window=window,
        num_policies=num_policies, hidden=args.hidden, 
        latent=args.latent, variant=args.variant
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    # Training loop
    os.makedirs(args.save_dir, exist_ok=True)
    results = {"epochs": args.epochs, "results": [], "variant": args.variant}
    
    best_val_ade = float('inf')
    
    for epoch in range(args.epochs):
        start_time = time.time()
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, args.beta, device)
        
        # Validate
        val_metrics = evaluate(model, val_loader, device)
        epoch_time = time.time() - start_time
        
        # Log
        result = {
            "epoch": epoch,
            "train_elbo": -train_loss,  # Convert to ELBO
            "val": val_metrics,
            "epoch_time_sec": epoch_time
        }
        results["results"].append(result)
        
        print(f"Epoch {epoch}/{args.epochs-1}: train_elbo={-train_loss:.4f}, "
              f"val_ADE={val_metrics['ADE']:.4f}, time={epoch_time:.1f}s")
        
        # Save best model
        if val_metrics["ADE"] < best_val_ade:
            best_val_ade = val_metrics["ADE"]
            torch.save(model.state_dict(), os.path.join(args.save_dir, "model_best.pt"))
    
    # Final test evaluation
    test_metrics = evaluate(model, test_loader, device)
    
    results.update({
        "num_policies": num_policies,
        "latent": args.latent,
        "hidden": args.hidden,
        "best_val": {"ADE": best_val_ade, "FDE": best_val_ade},
        "test": test_metrics
    })
    
    # Save results
    with open(os.path.join(args.save_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"Training completed. Best val ADE: {best_val_ade:.4f}")
    print(f"Test ADE: {test_metrics['ADE']:.4f}")


if __name__ == "__main__":
    main()
