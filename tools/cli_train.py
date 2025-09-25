from __future__ import annotations
import argparse
import os
import subprocess


def run(cmd: list[str], cwd: str | None = None):
    print("[ident-train] $", " ".join(cmd))
    result = subprocess.run(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    print(result.stdout)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def main():
    ap = argparse.ArgumentParser(description="Train IDENT baseline (state-conditional GRU)")
    ap.add_argument("--data_root", type=str, default="data/v1.0")
    ap.add_argument("--save_dir", type=str, default="runs/gru")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--gaussian_head", action="store_true")
    args = ap.parse_args()

    # Normalize device aliases
    if str(args.device).lower() == "gpu":
        args.device = "cuda"

    cmd = [
        "python", "baselines/state_cond/train_gru.py",
        "--data_root", args.data_root,
        "--save_dir", args.save_dir,
        "--device", args.device,
    ]
    if args.gaussian_head:
        cmd.append("--gaussian_head")

    repo_root = os.path.dirname(os.path.abspath(__file__)) + "/.."
    run(cmd, cwd=repo_root)


if __name__ == "__main__":
    main()


