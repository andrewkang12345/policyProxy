from __future__ import annotations
import argparse
import os
import subprocess


def run(cmd: list[str], cwd: str | None = None):
    print("[ident-eval] $", " ".join(cmd))
    result = subprocess.run(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    print(result.stdout)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def main():
    ap = argparse.ArgumentParser(description="Evaluate: rollout (action) + rep_similarity (representation)")
    ap.add_argument("--data_root", type=str, default="data/v1.0")
    ap.add_argument("--run_dir", type=str, default="runs/gru")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--no_visualize", action="store_true", help="Skip episode visualization")
    args = ap.parse_args()

    repo_root = os.path.dirname(os.path.abspath(__file__)) + "/.."
    model_pt = os.path.join(args.run_dir, "model_best.pt")

    os.makedirs(args.run_dir, exist_ok=True)

    # Normalize device aliases
    if str(args.device).lower() == "gpu":
        args.device = "cuda"

    # Drop intent probe; focus on action and representation

    # No episode visualization in streamlined eval

    # Rollout
    run(["python", "-m", "eval.rollout", "--data_root", args.data_root, "--model", model_pt, "--episodes", str(args.episodes), "--device", args.device,
         "--save_json", os.path.join(args.run_dir, "rollout.json"),
         "--save_plot", os.path.join(args.run_dir, "rollout.png")], cwd=repo_root)

    # Representation similarity plot
    rep_json = os.path.join(args.run_dir, "rep_similarity.json")
    run(["python", "-m", "eval.rep_similarity", "--data_root", args.data_root, "--save_json", rep_json, "--device", args.device, "--max_items", str(args.episodes * 1000), "--save_plot", args.run_dir], cwd=repo_root)


if __name__ == "__main__":
    main()


