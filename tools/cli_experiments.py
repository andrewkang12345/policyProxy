from __future__ import annotations
import argparse
import os
import subprocess
import json


def run(cmd: list[str], cwd: str | None = None):
    print("[ident-exp] $", " ".join(cmd))
    result = subprocess.run(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    print(result.stdout)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def write_config(base_cfg: str, out_cfg: str, policy_categories: list[str], oids: dict | None = None):
    import yaml
    with open(base_cfg, "r") as f:
        y = yaml.safe_load(f)
    y.pop("policies", None)
    y["policy_categories"] = policy_categories
    if oids is not None:
        y["oids"] = oids
    os.makedirs(os.path.dirname(out_cfg), exist_ok=True)
    with open(out_cfg, "w") as f:
        yaml.safe_dump(y, f)


def main():
    ap = argparse.ArgumentParser(description="Experiment analyzer: compute similarity and rollout summaries for existing data/model")
    ap.add_argument("--data_root", type=str, required=True, help="Root with train/test and optional ood_* splits")
    ap.add_argument("--run_dir", type=str, required=True, help="Directory containing model_best.pt and results.json from training")
    ap.add_argument("--device", type=str, default="gpu")
    ap.add_argument("--max_items", type=int, default=5000)
    args = ap.parse_args()

    repo_root = os.path.dirname(os.path.abspath(__file__)) + "/.."
    os.makedirs(args.run_dir, exist_ok=True)

    # Normalize device aliases
    if str(args.device).lower() == "gpu":
        args.device = "cuda"

    # Representation similarity against train for any available OOD splits
    sim_path = os.path.join(args.run_dir, "rep_similarity.json")
    run(["python", "-m", "eval.rep_similarity", "--data_root", args.data_root, "--save_json", sim_path, "--device", args.device, "--max_items", str(args.max_items), "--save_plot", args.run_dir], cwd=repo_root)

    # Diagnostics report (linear probe, clustering)
    run(["python", "-m", "eval.diagnostics", "--data_root", args.data_root, "--run_dir", args.run_dir, "--device", args.device, "--max_items", str(args.max_items)], cwd=repo_root)

    # Rollout summaries per available OOD split using best checkpoint
    model_pt = os.path.join(args.run_dir, "model_best.pt")
    # On IID
    run(["python", "-m", "eval.rollout", "--data_root", args.data_root, "--model", model_pt, "--episodes", "5", "--device", args.device,
         "--save_json", os.path.join(args.run_dir, "rollout_iid.json"),
         "--save_plot", os.path.join(args.run_dir, "rollout_iid.png")], cwd=repo_root)
    # On each OOD split
    for name in sorted(os.listdir(args.data_root)):
        if not name.startswith("ood_"):
            continue
        dr = os.path.join(args.data_root, name)
        run(["python", "-m", "eval.rollout", "--data_root", dr, "--model", model_pt, "--episodes", "5", "--device", args.device,
             "--save_json", os.path.join(args.run_dir, f"rollout_{name}.json"),
             "--save_plot", os.path.join(args.run_dir, f"rollout_{name}.png")], cwd=repo_root)


if __name__ == "__main__":
    main()


