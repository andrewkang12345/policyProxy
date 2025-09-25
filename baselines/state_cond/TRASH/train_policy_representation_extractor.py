#!/usr/bin/env python3
"""
Policy Representation Extractor for Task A2

Trains a model to extract policy representation vectors that can be used
as latent z in Fixed-Z CVAE for action output tasks.
"""

import argparse
import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from tasks.common.dataset import NextFrameDataset


class PolicyRepresentationExtractor(nn.Module):
    """Extract policy representations from state-action sequences."""
    
    def __init__(self, state_dim=2, action_dim=2, hidden_dim=128, repr_dim=16, num_policies=2):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.repr_dim = repr_dim
        self.num_policies = num_policies
        
        # Encoder: state-action -> hidden representation
        self.encoder = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # GRU for temporal modeling
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        
        # Policy representation head
        self.repr_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, repr_dim)
        )
        
        # Policy classifier (for training supervision)
        self.classifier = nn.Sequential(
            nn.Linear(repr_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_policies)
        )
        
    def forward(self, states, actions, return_sequence=False):
        """
        Extract policy representations.
        
        Args:
            states: [B, T, state_dim] state sequences
            actions: [B, T, action_dim] action sequences  
            return_sequence: If True, return representations for all timesteps
            
        Returns:
            policy_repr: [B, repr_dim] or [B, T, repr_dim] policy representations
            policy_logits: [B, num_policies] or [B, T, num_policies] classification logits
        """
        B, T = states.shape[:2]
        
        # Concatenate states and actions
        state_action = torch.cat([states, actions], dim=-1)  # [B, T, state_dim + action_dim]
        
        # Encode each timestep
        encoded = self.encoder(state_action.view(-1, self.state_dim + self.action_dim))  # [B*T, hidden_dim]
        encoded = encoded.view(B, T, self.hidden_dim)  # [B, T, hidden_dim]
        
        # Temporal modeling with GRU
        gru_out, hidden = self.gru(encoded)  # gru_out: [B, T, hidden_dim], hidden: [1, B, hidden_dim]
        
        if return_sequence:
            # Return representations for all timesteps
            policy_repr = self.repr_head(gru_out)  # [B, T, repr_dim]
            policy_logits = self.classifier(policy_repr)  # [B, T, num_policies]
        else:
            # Return final representation only
            final_hidden = hidden.squeeze(0)  # [B, hidden_dim]
            policy_repr = self.repr_head(final_hidden)  # [B, repr_dim]
            policy_logits = self.classifier(policy_repr)  # [B, num_policies]
            
        return policy_repr, policy_logits


def collate_with_policy_ids(batch):
    """Collate function that includes policy IDs."""
    states = []
    actions = []
    policy_ids = []
    
    for sample in batch:
        if isinstance(sample, dict):
            state = sample.get("state", sample.get("pos"))
            action = sample.get("action", sample.get("vel"))
            policy_id = sample.get("policy_id", sample.get("intent", 0))
        else:
            state, action = sample[0], sample[1]
            policy_id = 0  # Default policy ID
            
        if state is not None and action is not None:
            states.append(torch.tensor(state, dtype=torch.float32))
            actions.append(torch.tensor(action, dtype=torch.float32))
            policy_ids.append(policy_id)
    
    if not states:
        return None
        
    states = torch.stack(states)
    actions = torch.stack(actions)
    policy_ids = torch.tensor(policy_ids, dtype=torch.long)
    
    return states, actions, policy_ids


def train_epoch(model, train_loader, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    count = 0
    
    for batch in train_loader:
        if batch is None:
            continue
            
        states, actions, policy_ids = batch
        states = states.to(device)
        actions = actions.to(device)
        policy_ids = policy_ids.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        policy_repr, policy_logits = model(states, actions)
        
        # Classification loss
        loss = F.cross_entropy(policy_logits, policy_ids)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Metrics
        total_loss += loss.item()
        pred = policy_logits.argmax(dim=-1)
        total_acc += (pred == policy_ids).float().mean().item()
        count += 1
    
    return {
        "loss": total_loss / count if count > 0 else float('inf'),
        "accuracy": total_acc / count if count > 0 else 0.0
    }


def evaluate(model, val_loader, device):
    """Evaluate the model."""
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    count = 0
    
    with torch.no_grad():
        for batch in val_loader:
            if batch is None:
                continue
                
            states, actions, policy_ids = batch
            states = states.to(device)
            actions = actions.to(device)  
            policy_ids = policy_ids.to(device)
            
            # Forward pass
            policy_repr, policy_logits = model(states, actions)
            
            # Metrics
            loss = F.cross_entropy(policy_logits, policy_ids)
            total_loss += loss.item()
            pred = policy_logits.argmax(dim=-1)
            total_acc += (pred == policy_ids).float().mean().item()
            count += 1
    
    return {
        "loss": total_loss / count if count > 0 else float('inf'),
        "accuracy": total_acc / count if count > 0 else 0.0
    }


def main():
    parser = argparse.ArgumentParser(description='Train Policy Representation Extractor')
    parser.add_argument('--data_root', required=True, help='Path to dataset root')
    parser.add_argument('--save_dir', required=True, help='Directory to save model')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--hidden_dim', type=int, default=128, help='Hidden dimension')
    parser.add_argument('--repr_dim', type=int, default=16, help='Representation dimension')
    parser.add_argument('--num_policies', type=int, default=2, help='Number of policies')
    parser.add_argument('--device', default='cpu', help='Device to use')
    args = parser.parse_args()
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Device setup
    device = torch.device(args.device)
    
    # Load datasets
    train_index = os.path.join(args.data_root, "train", "index.json")
    val_index = os.path.join(args.data_root, "val", "index.json")
    
    if not os.path.exists(train_index):
        raise FileNotFoundError(f"No index.json found at {train_index}")
        
    train_dataset = NextFrameDataset(train_index)
    val_dataset = NextFrameDataset(val_index)
    
    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, 
                             collate_fn=collate_with_policy_ids)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                           collate_fn=collate_with_policy_ids)
    
    # Model
    model = PolicyRepresentationExtractor(
        state_dim=2,  # position
        action_dim=2,  # velocity 
        hidden_dim=args.hidden_dim,
        repr_dim=args.repr_dim,
        num_policies=args.num_policies
    ).to(device)
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Training loop
    best_acc = 0.0
    results = {"train": [], "val": []}
    
    print(f"Training Policy Representation Extractor for {args.epochs} epochs...")
    
    for epoch in range(args.epochs):
        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, device)
        
        # Validate
        val_metrics = evaluate(model, val_loader, device)
        
        # Save metrics
        results["train"].append(train_metrics)
        results["val"].append(val_metrics)
        
        # Save best model
        if val_metrics["accuracy"] > best_acc:
            best_acc = val_metrics["accuracy"]
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'val_accuracy': val_metrics["accuracy"],
                'args': vars(args)
            }, os.path.join(args.save_dir, "model_best.pt"))
        
        print(f"Epoch {epoch}/{args.epochs-1}: "
              f"train_loss={train_metrics['loss']:.4f}, train_acc={train_metrics['accuracy']:.4f}, "
              f"val_loss={val_metrics['loss']:.4f}, val_acc={val_metrics['accuracy']:.4f}")
    
    # Save results
    with open(os.path.join(args.save_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"Training completed. Best val accuracy: {best_acc:.4f}")
    print(f"Model saved to {args.save_dir}")


if __name__ == "__main__":
    main()
