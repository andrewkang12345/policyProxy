#!/usr/bin/env python3
"""
Policy Classification Evaluation (Task B1)

Evaluates models on the task of classifying ground truth policy IDs.
This is a segment-aware task where the model knows the fixed policy segments.
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
from sklearn.metrics import accuracy_score, f1_score, classification_report

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from tasks.common.dataset import NextFrameDataset


class PolicyClassifier(nn.Module):
    """Simple policy classifier for evaluation."""
    
    def __init__(self, input_dim=2, hidden_dim=128, num_policies=2, window=6):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_policies = num_policies
        self.window = window
        
        # Feature encoder - handle variable input sizes
        # Maximum possible input size for flexible architecture
        max_input_size = input_dim * window * 3  # Assume max 3 agents
        
        self.input_projection = nn.Linear(max_input_size, hidden_dim)
        
        self.encoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_policies)
        )
        
    def forward(self, x):
        """
        Classify policy from input features.
        
        Args:
            x: [B, window, input_dim] input features
            
        Returns:
            logits: [B, num_policies] classification logits
        """
        # Handle different input dimensionalities
        if len(x.shape) == 5:
            # [B, W, T, A, D] -> flatten everything after batch
            B = x.shape[0]
            x_flat = x.view(B, -1)
        elif len(x.shape) == 4:
            B, W, A, D = x.shape
            x_flat = x.view(B, W * A * D)
        elif len(x.shape) == 3:
            B, W, D = x.shape
            x_flat = x.view(B, W * D)
        else:
            raise ValueError(f"Unexpected input shape: {x.shape}")  # Flatten window
        
        # Pad to maximum size if needed
        max_size = self.input_dim * self.window * 3
        if x_flat.size(1) < max_size:
            import torch
            padding = torch.zeros(x_flat.size(0), max_size - x_flat.size(1), device=x_flat.device)
            x_flat = torch.cat([x_flat, padding], dim=1)
        elif x_flat.size(1) > max_size:
            x_flat = x_flat[:, :max_size]  # Truncate if too large
        
        # Project to hidden dimension
        projected = self.input_projection(x_flat)
        features = self.encoder(projected)
        logits = self.classifier(features)
        
        return logits


def collate_policy_classification(batch):
    """Collate function for policy classification."""
    states = []
    policy_ids = []
    
    for sample in batch:
        if isinstance(sample, dict):
            state = sample.get("state", sample.get("pos"))
            policy_id = sample.get("policy_id", sample.get("intent", 0))
        else:
            state = sample[0]
            policy_id = 0  # Default policy ID
            
        if state is not None:
            states.append(torch.tensor(state, dtype=torch.float32))
            policy_ids.append(policy_id)
    
    if not states:
        return None
        
    states = torch.stack(states)
    policy_ids = torch.tensor(policy_ids, dtype=torch.long)
    
    return states, policy_ids


def train_classifier(model, train_loader, val_loader, device, epochs=20, lr=1e-3):
    """Train the policy classifier."""
    optimizer = optim.Adam(model.parameters(), lr=lr)
    best_acc = 0.0
    
    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch in train_loader:
            if batch is None:
                continue
                
            states, policy_ids = batch
            states = states.to(device)
            policy_ids = policy_ids.to(device)
            
            optimizer.zero_grad()
            logits = model(states)
            loss = F.cross_entropy(logits, policy_ids)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            pred = logits.argmax(dim=-1)
            train_correct += (pred == policy_ids).sum().item()
            train_total += policy_ids.size(0)
        
        # Validate
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                if batch is None:
                    continue
                    
                states, policy_ids = batch
                states = states.to(device)
                policy_ids = policy_ids.to(device)
                
                logits = model(states)
                loss = F.cross_entropy(logits, policy_ids)
                
                val_loss += loss.item()
                pred = logits.argmax(dim=-1)
                val_correct += (pred == policy_ids).sum().item()
                val_total += policy_ids.size(0)
        
        train_acc = train_correct / train_total if train_total > 0 else 0.0
        val_acc = val_correct / val_total if val_total > 0 else 0.0
        
        if val_acc > best_acc:
            best_acc = val_acc
        
        if epoch % 5 == 0:
            print(f"Epoch {epoch}: train_acc={train_acc:.4f}, val_acc={val_acc:.4f}")
    
    return best_acc


def evaluate_classifier(model, test_loader, device):
    """Evaluate the trained classifier."""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            if batch is None:
                continue
                
            states, policy_ids = batch
            states = states.to(device)
            policy_ids = policy_ids.to(device)
            
            logits = model(states)
            pred = logits.argmax(dim=-1)
            
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(policy_ids.cpu().numpy())
    
    if not all_preds:
        return {"accuracy": 0.0, "f1_score": 0.0}
    
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    
    return {
        "accuracy": accuracy,
        "f1_score": f1,
        "classification_report": classification_report(all_labels, all_preds, output_dict=True)
    }


def main():
    parser = argparse.ArgumentParser(description='Policy Classification Evaluation (Task B1)')
    parser.add_argument('--data_root', required=True, help='Path to dataset root')
    parser.add_argument('--model_type', default='policy_classifier', help='Model type')
    parser.add_argument('--save_dir', required=True, help='Directory to save results')
    parser.add_argument('--segment_aware', action='store_true', help='Use segment-aware evaluation')
    parser.add_argument('--task', default='policy_classification', help='Task name')
    parser.add_argument('--epochs', type=int, default=20, help='Training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--device', default='cpu', help='Device to use')
    parser.add_argument('--model', help='Pretrained model path (optional)')
    args = parser.parse_args()
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Device setup
    device = torch.device(args.device)
    
    # Load datasets
    if os.path.isdir(args.data_root):
        # Training mode - load train/val/test splits
        train_index = os.path.join(args.data_root, "train", "index.json")
        val_index = os.path.join(args.data_root, "val", "index.json")
        test_index = os.path.join(args.data_root, "test", "index.json")
        
        if os.path.exists(train_index):
            train_dataset = NextFrameDataset(train_index)
            val_dataset = NextFrameDataset(val_index) if os.path.exists(val_index) else train_dataset
            test_dataset = NextFrameDataset(test_index) if os.path.exists(test_index) else train_dataset
            
            train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                     collate_fn=collate_policy_classification)
            val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                                   collate_fn=collate_policy_classification)
            test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                                    collate_fn=collate_policy_classification)
            
            mode = "train"
        else:
            # Single split evaluation
            index_path = os.path.join(args.data_root, "index.json")
            if os.path.exists(index_path):
                dataset = NextFrameDataset(index_path)
                test_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                                        collate_fn=collate_policy_classification)
                mode = "eval"
            else:
                raise FileNotFoundError(f"No dataset found at {args.data_root}")
    else:
        raise FileNotFoundError(f"Invalid data root: {args.data_root}")
    
    # Model
    model = PolicyClassifier(
        input_dim=2,  # positions
        hidden_dim=128,
        num_policies=2,
        window=6
    ).to(device)
    
    if mode == "train":
        # Training mode
        print(f"Training policy classifier for {args.epochs} epochs...")
        best_acc = train_classifier(model, train_loader, val_loader, device, 
                                   epochs=args.epochs, lr=1e-3)
        
        # Save trained model
        torch.save({
            'model_state_dict': model.state_dict(),
            'best_accuracy': best_acc,
            'args': vars(args)
        }, os.path.join(args.save_dir, "model_best.pt"))
        
        # Evaluate on test set
        test_results = evaluate_classifier(model, test_loader, device)
        print(f"Test Results: accuracy={test_results['accuracy']:.4f}, f1={test_results['f1_score']:.4f}")
        
        # Save results
        results = {
            "task": "policy_classification_segment_aware" if args.segment_aware else "policy_classification",
            "best_train_accuracy": best_acc,
            "test_results": test_results,
            "segment_aware": args.segment_aware
        }
        
    else:
        # Evaluation mode
        if args.model and os.path.exists(args.model):
            # Load pretrained model
            checkpoint = torch.load(args.model, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded model from {args.model}")
        
        # Evaluate
        test_results = evaluate_classifier(model, test_loader, device)
        print(f"Evaluation Results: accuracy={test_results['accuracy']:.4f}, f1={test_results['f1_score']:.4f}")
        
        results = {
            "task": "policy_classification_segment_aware" if args.segment_aware else "policy_classification",
            "test_results": test_results,
            "segment_aware": args.segment_aware,
            "split": os.path.basename(args.data_root)
        }
    
    # Save results
    with open(os.path.join(args.save_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {args.save_dir}")


if __name__ == "__main__":
    main()
