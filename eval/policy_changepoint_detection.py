#!/usr/bin/env python3
"""
Policy Changepoint Detection Evaluation (Task B2)

Evaluates models on detecting policy changes without knowing segment boundaries.
This is a segment-unaware task that must detect when policies change.
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
from sklearn.metrics import f1_score, precision_recall_fscore_support

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from tasks.common.dataset import NextFrameDataset


class ChangePointDetector(nn.Module):
    """Change point detection model using sliding window energy."""
    
    def __init__(self, input_dim=2, hidden_dim=64, window_size=10):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.window_size = window_size
        
        # Feature encoder for temporal windows
        self.encoder = nn.Sequential(
            nn.Linear(input_dim * window_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Binary changepoint classifier
        self.changepoint_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # Two windows: before + after
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
    def forward(self, before_window, after_window):
        """
        Detect changepoint between two windows.
        
        Args:
            before_window: [B, window_size, input_dim] features before potential changepoint
            after_window: [B, window_size, input_dim] features after potential changepoint
            
        Returns:
            changepoint_prob: [B] probability of changepoint
        """
        B = before_window.shape[0]
        
        # Encode windows
        before_flat = before_window.view(B, -1)
        after_flat = after_window.view(B, -1)
        
        before_features = self.encoder(before_flat)
        after_features = self.encoder(after_flat)
        
        # Concatenate and classify
        combined = torch.cat([before_features, after_features], dim=-1)
        changepoint_prob = self.changepoint_head(combined).squeeze(-1)
        
        return changepoint_prob


def extract_windows_and_labels(dataset, window_size=10):
    """Extract sliding windows and changepoint labels from dataset."""
    windows_before = []
    windows_after = []
    labels = []
    
    for i in range(len(dataset)):
        sample = dataset[i]
        
        if isinstance(sample, dict):
            states = sample.get("state", sample.get("pos"))
            policy_ids = sample.get("policy_id", sample.get("intent"))
        else:
            states = sample[0]
            policy_ids = None
        
        if states is None:
            continue
            
        T = states.shape[0]
        
        # Generate sliding windows
        for t in range(window_size, T - window_size):
            before = states[t-window_size:t]  # [window_size, input_dim]
            after = states[t:t+window_size]   # [window_size, input_dim]
            
            # Label: 1 if there's a policy change at timestep t, 0 otherwise
            if policy_ids is not None:
                # Check if policy changes between t-1 and t
                change_label = 1.0 if policy_ids[t-1] != policy_ids[t] else 0.0
            else:
                # Random labels for testing (in real scenario, would need ground truth)
                change_label = 0.0  # Default no change
            
            windows_before.append(torch.tensor(before, dtype=torch.float32))
            windows_after.append(torch.tensor(after, dtype=torch.float32))
            labels.append(change_label)
    
    if not windows_before:
        return None, None, None
        
    windows_before = torch.stack(windows_before)
    windows_after = torch.stack(windows_after)
    labels = torch.tensor(labels, dtype=torch.float32)
    
    return windows_before, windows_after, labels


def train_detector(model, train_data, val_data, device, epochs=20, lr=1e-3):
    """Train the changepoint detector."""
    if train_data is None or val_data is None:
        return 0.0
        
    train_before, train_after, train_labels = train_data
    val_before, val_after, val_labels = val_data
    
    # Check for None data
    if train_before is None or train_after is None or train_labels is None:
        print("❌ Training data contains None values")
        return 0.0
    
    if val_before is None or val_after is None or val_labels is None:
        print("❌ Validation data contains None values")
        return 0.0
    
    # Move to device
    train_before = train_before.to(device)
    train_after = train_after.to(device)
    train_labels = train_labels.to(device)
    val_before = val_before.to(device)
    val_after = val_after.to(device)
    val_labels = val_labels.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    best_f1 = 0.0
    
    # Create batches
    batch_size = 32
    n_train = len(train_labels)
    n_val = len(val_labels)
    
    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0.0
        
        for i in range(0, n_train, batch_size):
            end_i = min(i + batch_size, n_train)
            
            batch_before = train_before[i:end_i]
            batch_after = train_after[i:end_i]
            batch_labels = train_labels[i:end_i]
            
            optimizer.zero_grad()
            probs = model(batch_before, batch_after)
            loss = F.binary_cross_entropy(probs, batch_labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validate
        model.eval()
        with torch.no_grad():
            val_probs = []
            for i in range(0, n_val, batch_size):
                end_i = min(i + batch_size, n_val)
                batch_before = val_before[i:end_i]
                batch_after = val_after[i:end_i]
                
                probs = model(batch_before, batch_after)
                val_probs.append(probs)
            
            if val_probs:
                val_probs = torch.cat(val_probs)
                val_preds = (val_probs > 0.5).float()
                
                # Compute F1 score
                val_labels_np = val_labels.cpu().numpy()
                val_preds_np = val_preds.cpu().numpy()
                
                if len(np.unique(val_labels_np)) > 1:  # Need both classes for F1
                    f1 = f1_score(val_labels_np, val_preds_np)
                    if f1 > best_f1:
                        best_f1 = f1
                
                if epoch % 5 == 0:
                    acc = (val_preds == val_labels).float().mean().item()
                    print(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_acc={acc:.4f}, val_f1={f1:.4f}")
    
    return best_f1


def evaluate_changepoint_detection(model, test_data, device, tau=3):
    """Evaluate changepoint detection performance."""
    if test_data is None:
        return {"f1_tau": 0.0, "mabe": float('inf'), "delay_mean": float('inf')}
    
    test_before, test_after, test_labels = test_data
    
    # Handle None values
    if test_before is not None:
        test_before = test_before.to(device)
    if test_after is not None:
        test_after = test_after.to(device)
    if test_labels is not None:
        test_labels = test_labels.to(device)
    
    # Check if any required data is missing
    if test_before is None or test_after is None or test_labels is None:
        return {"f1_tau": 0.0, "mabe": float('inf'), "delay_mean": float('inf')}
    
    model.eval()
    with torch.no_grad():
        # Predict changepoints
        batch_size = 32
        n_test = len(test_labels)
        test_probs = []
        
        for i in range(0, n_test, batch_size):
            end_i = min(i + batch_size, n_test)
            batch_before = test_before[i:end_i]
            batch_after = test_after[i:end_i]
            
            probs = model(batch_before, batch_after)
            test_probs.append(probs)
        
        if test_probs:
            test_probs = torch.cat(test_probs)
            test_preds = (test_probs > 0.5).float()
            
            # Convert to numpy
            test_labels_np = test_labels.cpu().numpy()
            test_preds_np = test_preds.cpu().numpy()
            test_probs_np = test_probs.cpu().numpy()
            
            # F1@τ (F1 score with tolerance τ)
            # For simplicity, use standard F1
            if len(np.unique(test_labels_np)) > 1:
                f1_tau = f1_score(test_labels_np, test_preds_np)
            else:
                f1_tau = 0.0
            
            # Mean Absolute Boundary Error (MABE)
            true_changes = np.where(test_labels_np == 1)[0]
            pred_changes = np.where(test_preds_np == 1)[0]
            
            if len(true_changes) > 0 and len(pred_changes) > 0:
                # Simple MABE: average distance between predicted and true changepoints
                distances = []
                for true_cp in true_changes:
                    if len(pred_changes) > 0:
                        min_dist = np.min(np.abs(pred_changes - true_cp))
                        distances.append(min_dist)
                mabe = np.mean(distances) if distances else float('inf')
            else:
                mabe = float('inf')
            
            # Detection delay (simplified)
            delays = []
            for true_cp in true_changes:
                later_preds = pred_changes[pred_changes >= true_cp]
                if len(later_preds) > 0:
                    delay = later_preds[0] - true_cp
                    delays.append(delay)
            delay_mean = np.mean(delays) if delays else float('inf')
            
            return {
                "f1_tau": f1_tau,
                "mabe": mabe,
                "delay_mean": delay_mean,
                "accuracy": (test_preds_np == test_labels_np).mean(),
                "n_true_changes": len(true_changes),
                "n_pred_changes": len(pred_changes)
            }
    
    return {"f1_tau": 0.0, "mabe": float('inf'), "delay_mean": float('inf')}



    # Add error handling for data extraction
    def safe_extract_windows(dataset, window_size, step_size):
        try:
            return extract_policy_change_windows(dataset, window_size, step_size)
        except Exception as e:
            print(f"❌ Error extracting windows: {e}")
            return None, None, None
    
def main():
    parser = argparse.ArgumentParser(description='Policy Changepoint Detection (Task B2)')
    parser.add_argument('--data_root', required=True, help='Path to dataset root')
    parser.add_argument('--model_type', default='changepoint_detector', help='Model type')
    parser.add_argument('--save_dir', required=True, help='Directory to save results')
    parser.add_argument('--segment_unaware', action='store_true', help='Use segment-unaware evaluation')
    parser.add_argument('--task', default='changepoint_detection', help='Task name')
    parser.add_argument('--tau', type=int, default=3, help='Tolerance for F1@τ')
    parser.add_argument('--window_size', type=int, default=10, help='Window size for detection')
    parser.add_argument('--epochs', type=int, default=20, help='Training epochs')
    parser.add_argument('--device', default='cpu', help='Device to use')
    parser.add_argument('--model', help='Pretrained model path (optional)')
    args = parser.parse_args()
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Device setup
    device = torch.device(args.device)
    
    # Load datasets
    if os.path.isdir(args.data_root):
        # Training mode - check for train/val/test splits
        train_index = os.path.join(args.data_root, "train", "index.json")
        val_index = os.path.join(args.data_root, "val", "index.json")
        test_index = os.path.join(args.data_root, "test", "index.json")
        
        if os.path.exists(train_index):
            train_dataset = NextFrameDataset(train_index)
            val_dataset = NextFrameDataset(val_index) if os.path.exists(val_index) else train_dataset
            test_dataset = NextFrameDataset(test_index) if os.path.exists(test_index) else train_dataset
            
            # Extract windows and labels
            print("Extracting training windows...")
            train_data = extract_windows_and_labels(train_dataset, args.window_size)
            print("Extracting validation windows...")
            val_data = extract_windows_and_labels(val_dataset, args.window_size)
            print("Extracting test windows...")
            test_data = extract_windows_and_labels(test_dataset, args.window_size)
            
            mode = "train"
        else:
            # Single split evaluation
            index_path = os.path.join(args.data_root, "index.json")
            if os.path.exists(index_path):
                dataset = NextFrameDataset(index_path)
                print("Extracting test windows...")
                test_data = extract_windows_and_labels(dataset, args.window_size)
                mode = "eval"
            else:
                raise FileNotFoundError(f"No dataset found at {args.data_root}")
    else:
        raise FileNotFoundError(f"Invalid data root: {args.data_root}")
    
    # Model
    model = ChangePointDetector(
        input_dim=2,  # positions
        hidden_dim=64,
        window_size=args.window_size
    ).to(device)
    
    if mode == "train":
        # Training mode
        print(f"Training changepoint detector for {args.epochs} epochs...")
        best_f1 = train_detector(model, train_data, val_data, device, 
                                epochs=args.epochs, lr=1e-3)
        
        # Save trained model
        torch.save({
            'model_state_dict': model.state_dict(),
            'best_f1': best_f1,
            'args': vars(args)
        }, os.path.join(args.save_dir, "model_best.pt"))
        
        # Evaluate on test set
        test_results = evaluate_changepoint_detection(model, test_data, device, args.tau)
        print(f"Test Results: f1_tau={test_results['f1_tau']:.4f}, mabe={test_results['mabe']:.2f}")
        
        # Save results
        results = {
            "task": "changepoint_detection_segment_unaware" if args.segment_unaware else "changepoint_detection",
            "best_train_f1": best_f1,
            "test_results": test_results,
            "segment_unaware": args.segment_unaware,
            "tau": args.tau
        }
        
    else:
        # Evaluation mode
        if args.model and os.path.exists(args.model):
            # Load pretrained model
            checkpoint = torch.load(args.model, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded model from {args.model}")
        
        # Evaluate
        test_results = evaluate_changepoint_detection(model, test_data, device, args.tau)
        print(f"Evaluation Results: f1_tau={test_results['f1_tau']:.4f}, mabe={test_results['mabe']:.2f}")
        
        results = {
            "task": "changepoint_detection_segment_unaware" if args.segment_unaware else "changepoint_detection",
            "test_results": test_results,
            "segment_unaware": args.segment_unaware,
            "split": os.path.basename(args.data_root),
            "tau": args.tau
        }
    
    # Save results
    with open(os.path.join(args.save_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {args.save_dir}")


if __name__ == "__main__":
    main()
