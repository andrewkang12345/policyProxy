#!/usr/bin/env python3
"""
Policy Divergence Computation for Categorical Policy IDs

Computes proper categorical divergences between policy distributions
instead of using Wasserstein distance on policy IDs.
"""

import argparse
import os
import json
import numpy as np
from collections import Counter
from typing import Dict, List, Tuple, Any
from scipy.spatial.distance import jensenshannon
from scipy.stats import entropy

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from tasks.common.dataset import NextFrameDataset


def extract_policy_ids(dataset_path: str) -> List[int]:
    """Extract policy IDs from a dataset."""
    try:
        dataset = NextFrameDataset(dataset_path)
        policy_ids = []
    except (IsADirectoryError, FileNotFoundError):
        # Handle case where dataset_path is the directory containing train/val/test
        index_path = os.path.join(dataset_path, "index.json")
        if os.path.exists(index_path):
            dataset = NextFrameDataset(dataset_path)
            policy_ids = []
        else:
            return []
    
    for ep in dataset.episodes:
        # Try different field names
        ids = ep.get("policy_ids")
        if ids is None:
            ids = ep.get("intents")
        if ids is None:
            ids = ep.get("policy_id")
        if ids is None:
            continue
            
        if isinstance(ids, (int, float)):
            policy_ids.append(int(ids))
        elif isinstance(ids, (list, np.ndarray)):
            # Take most common policy ID in episode
            if len(ids) > 0:
                if isinstance(ids, np.ndarray):
                    ids = ids.flatten()
                most_common = Counter(ids).most_common(1)
                if most_common:
                    policy_ids.append(int(most_common[0][0]))
    
    return policy_ids


def get_policy_distribution(policy_ids: List[int], num_policies: int = None) -> np.ndarray:
    """Convert policy IDs to probability distribution."""
    if not policy_ids:
        return np.array([])
    
    if num_policies is None:
        num_policies = max(policy_ids) + 1
    
    # Count occurrences
    counts = np.bincount(policy_ids, minlength=num_policies)
    
    # Convert to probabilities
    prob_dist = counts / np.sum(counts)
    return prob_dist


def compute_categorical_divergences(train_dist: np.ndarray, test_dist: np.ndarray) -> Dict[str, float]:
    """Compute various categorical divergences."""
    
    # Ensure same length
    max_len = max(len(train_dist), len(test_dist))
    if len(train_dist) < max_len:
        train_dist = np.pad(train_dist, (0, max_len - len(train_dist)))
    if len(test_dist) < max_len:
        test_dist = np.pad(test_dist, (0, max_len - len(test_dist)))
    
    # Add small epsilon to avoid log(0)
    eps = 1e-10
    train_dist = train_dist + eps
    test_dist = test_dist + eps
    
    # Renormalize
    train_dist = train_dist / np.sum(train_dist)
    test_dist = test_dist / np.sum(test_dist)
    
    divergences = {}
    
    # KL Divergence
    kl_div = entropy(test_dist, train_dist)
    divergences["kl_divergence"] = float(kl_div)
    
    # Jensen-Shannon Divergence
    js_div = jensenshannon(train_dist, test_dist)**2
    divergences["js_divergence"] = float(js_div)
    
    # Total Variation Distance
    tv_distance = 0.5 * np.sum(np.abs(train_dist - test_dist))
    divergences["tv_distance"] = float(tv_distance)
    
    # Chi-squared divergence
    chi2_div = np.sum((test_dist - train_dist)**2 / train_dist)
    divergences["chi2_divergence"] = float(chi2_div)
    
    # Hellinger distance
    hellinger = np.sqrt(0.5 * np.sum((np.sqrt(train_dist) - np.sqrt(test_dist))**2))
    divergences["hellinger_distance"] = float(hellinger)
    
    return divergences


def compute_policy_category_metrics(policy_ids: List[int]) -> Dict[str, Any]:
    """Compute policy category characteristics."""
    if not policy_ids:
        return {}
    
    unique_policies = set(policy_ids)
    num_unique = len(unique_policies)
    
    # Policy distribution entropy (higher = more diverse)
    dist = get_policy_distribution(policy_ids)
    policy_entropy = entropy(dist + 1e-10)
    
    # Policy switching rate (transitions between different policies)
    switches = 0
    for i in range(1, len(policy_ids)):
        if policy_ids[i] != policy_ids[i-1]:
            switches += 1
    switch_rate = switches / (len(policy_ids) - 1) if len(policy_ids) > 1 else 0
    
    return {
        "num_unique_policies": num_unique,
        "total_samples": len(policy_ids),
        "policy_entropy": float(policy_entropy),
        "policy_switch_rate": float(switch_rate),
        "policy_distribution": dist.tolist()
    }


def main():
    parser = argparse.ArgumentParser(description="Compute categorical policy divergences")
    parser.add_argument("--data_root", type=str, required=True, help="Path to dataset root")
    parser.add_argument("--save_dir", type=str, required=True, help="Directory to save results")
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Load train set as reference
    train_path = os.path.join(args.data_root, "train", "index.json")
    if not os.path.exists(train_path):
        print(f"❌ Train dataset not found: {train_path}")
        return
    
    print("📊 Computing categorical policy divergences...")
    
    # Extract train policy distribution
    train_policy_ids = extract_policy_ids(train_path)
    if not train_policy_ids:
        print("❌ No policy IDs found in training data")
        return
    
    train_dist = get_policy_distribution(train_policy_ids)
    train_metrics = compute_policy_category_metrics(train_policy_ids)
    num_policies = len(train_dist)
    
    print(f"✅ Found {len(train_policy_ids)} samples with {num_policies} unique policies in train set")
    
    # Process all splits
    all_results = {}
    
    # List all directories in data_root
    for split_name in os.listdir(args.data_root):
        split_path = os.path.join(args.data_root, split_name)
        if not os.path.isdir(split_path):
            continue
        
        # Check if it's a valid dataset split (has index.json)
        split_index = os.path.join(split_path, "index.json")
        if not os.path.exists(split_index):
            continue
        
        print(f"Processing {split_name}...")
        
        # Extract policy IDs
        split_policy_ids = extract_policy_ids(split_index)
        if not split_policy_ids:
            print(f"⚠️  No policy IDs found in {split_name}")
            continue
        
        # Get distribution
        split_dist = get_policy_distribution(split_policy_ids, num_policies)
        split_metrics = compute_policy_category_metrics(split_policy_ids)
        
        # Compute divergences vs train
        if split_name == "train":
            # Self-divergence should be 0
            divergences = {
                "kl_divergence": 0.0,
                "js_divergence": 0.0,
                "tv_distance": 0.0,
                "chi2_divergence": 0.0,
                "hellinger_distance": 0.0
            }
        else:
            divergences = compute_categorical_divergences(train_dist, split_dist)
        
        # Store results
        result = {
            "split": split_name,
            "policy_divergences": divergences,
            "policy_metrics": split_metrics,
            "reference": "train" if split_name != "train" else "self"
        }
        
        all_results[split_name] = result
        
        # Save individual result
        result_file = os.path.join(args.save_dir, f"policy_divergences_{split_name}.json")
        with open(result_file, "w") as f:
            json.dump(result, f, indent=2)
        
        print(f"  - KL divergence: {divergences['kl_divergence']:.4f}")
        print(f"  - JS divergence: {divergences['js_divergence']:.4f}")
        print(f"  - TV distance: {divergences['tv_distance']:.4f}")
    
    # Save aggregated results
    summary = {
        "experiment_info": {
            "data_root": args.data_root,
            "num_policies": num_policies,
            "train_samples": len(train_policy_ids)
        },
        "train_reference": {
            "policy_distribution": train_dist.tolist(),
            "metrics": train_metrics
        },
        "split_results": all_results
    }
    
    summary_file = os.path.join(args.save_dir, "policy_divergences_summary.json")
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"✅ Policy divergence analysis completed")
    print(f"📁 Results saved to {args.save_dir}")


if __name__ == "__main__":
    main()
