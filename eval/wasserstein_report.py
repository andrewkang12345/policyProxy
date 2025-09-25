from __future__ import annotations
import argparse
import json
import os
from typing import Dict

import yaml

from tools.distribution import collect_samples, SampleOptions, wasserstein_distance


def compute_wasserstein(train_index: str, split_index: str, opts: SampleOptions) -> Dict[str, float]:
    stats = {}
    base_state = collect_samples(train_index, "state", opts)
    split_state = collect_samples(split_index, "state", opts)
    stats["ws_state"] = float(wasserstein_distance(split_state, base_state))

    base_action = collect_samples(train_index, "action", opts)
    split_action = collect_samples(split_index, "action", opts)
    stats["ws_action"] = float(wasserstein_distance(split_action, base_action))

    base_policy = collect_samples(train_index, "policy", opts)
    split_policy = collect_samples(split_index, "policy", opts)
    stats["ws_policy"] = float(wasserstein_distance(split_policy, base_policy))
    return stats


def load_sample_options(data_root: str, limit: int) -> SampleOptions:
    cfg_path = os.path.join(data_root, "config_used.yaml")
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"config_used.yaml not found in {data_root}")
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
    gen = cfg.get("generator", {})
    arena = gen.get("arena", {})
    window = int(gen.get("window", 6))
    selected_team = int(gen.get("selected_team", 0))
    arena_size = (float(arena.get("width", 20.0)), float(arena.get("height", 14.0)))
    return SampleOptions(
        window=window,
        arena_size=arena_size,
        selected_team=selected_team,
        limit_episodes=limit,
    )


def main():
    ap = argparse.ArgumentParser(description="Compute Wasserstein divergences for dataset splits")
    ap.add_argument("--data_root", type=str, required=True, help="Root directory containing splits with index.json")
    ap.add_argument("--save_dir", type=str, required=True, help="Directory to save JSON summaries")
    ap.add_argument("--limit", type=int, default=32)
    args = ap.parse_args()

    train_index = os.path.join(args.data_root, "train", "index.json")
    if not os.path.exists(train_index):
        raise FileNotFoundError(f"Train index not found at {train_index}")

    os.makedirs(args.save_dir, exist_ok=True)

    opts = load_sample_options(args.data_root, args.limit)

    splits = ["train", "val", "test"]
    for name in os.listdir(args.data_root):
        if name.startswith("ood_"):
            splits.append(name)

    for split in splits:
        index_path = os.path.join(args.data_root, split, "index.json")
        if not os.path.exists(index_path):
            continue
        stats = compute_wasserstein(train_index, index_path, opts)
        out_path = os.path.join(args.save_dir, f"divergences_{split}.json")
        with open(out_path, "w") as f:
            json.dump({"split": split, "divergences": stats}, f, indent=2)
        print(json.dumps({"split": split, "divergences": stats}, indent=2))


if __name__ == "__main__":
    main()
