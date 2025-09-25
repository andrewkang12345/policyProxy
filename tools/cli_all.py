from __future__ import annotations
import argparse
import os
import subprocess


def run(cmd: list[str], cwd: str | None = None):
    print("[ident-all] $", " ".join(cmd))
    result = subprocess.run(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    print(result.stdout)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def main():
    ap = argparse.ArgumentParser(description="One command: data -> train -> eval -> visualize")
    ap.add_argument("--config", type=str, default="configs/matched_marginal.yaml")
    ap.add_argument("--out", type=str, default="data/v1.0")
    ap.add_argument("--run_dir", type=str, default="runs/gru_quickstart")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--episodes", type=int, default=3)
    group = ap.add_mutually_exclusive_group()
    group.add_argument("--full", action="store_true", help="Generate full dataset")
    group.add_argument("--tiny", action="store_true", help="Generate tiny demo dataset (default)")
    ap.add_argument("--skip_data", action="store_true")
    # HDF5 export removed for reproducibility simplicity
    ap.add_argument("--no_visualize", action="store_true", help="Skip episode visualization")
    args = ap.parse_args()

    repo_root = os.path.dirname(os.path.abspath(__file__)) + "/.."

    # Normalize device aliases
    if str(args.device).lower() == "gpu":
        args.device = "cuda"

    # Data
    if not args.skip_data:
        cmd = ["python", "make_data.py", "--config", args.config, "--out", args.out]
        if not args.full:
            cmd.append("--tiny")
        run(cmd, cwd=repo_root)

    # Train
    run(["python", "baselines/state_cond/train_gru.py", "--data_root", args.out, "--save_dir", args.run_dir, "--device", args.device], cwd=repo_root)

    # Evaluate bundle (action soundness + representation)
    model_pt = os.path.join(args.run_dir, "model_best.pt")
    # Intent probe & visualization removed
    te0 = os.path.join(args.out, "test", "ep_00000.npz")
    # Counterfactual removed
    run(["python", "-m", "eval.rollout", "--data_root", args.out, "--model", model_pt, "--episodes", str(args.episodes), "--device", args.device,
         "--save_json", os.path.join(args.run_dir, "rollout.json"),
         "--save_plot", os.path.join(args.run_dir, "rollout.png")], cwd=repo_root)
    # Diagnostics
    run(["python", "-m", "eval.diagnostics", "--data_root", args.out, "--run_dir", args.run_dir, "--device", args.device, "--max_items", "5000"], cwd=repo_root)
    # Representation similarity plot
    run(["python", "-m", "eval.rep_similarity", "--data_root", args.out, "--save_json", os.path.join(args.run_dir, "rep_similarity.json"), "--device", args.device, "--save_plot", args.run_dir], cwd=repo_root)

    print("[ident-all] Done. Artifacts in:", args.run_dir)


if __name__ == "__main__":
    main()


