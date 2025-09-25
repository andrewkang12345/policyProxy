from __future__ import annotations
import argparse
import os
import subprocess


def run(cmd: list[str], cwd: str | None = None):
    print("[ident-data] $", " ".join(cmd))
    result = subprocess.run(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    print(result.stdout)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def main():
    ap = argparse.ArgumentParser(description="Generate IDENT dataset (tiny or full)")
    ap.add_argument("--config", type=str, default="configs/matched_marginal.yaml")
    ap.add_argument("--out", type=str, default="data/v1.0")
    group = ap.add_mutually_exclusive_group()
    group.add_argument("--full", action="store_true", help="Generate full dataset")
    group.add_argument("--tiny", action="store_true", help="Generate tiny demo dataset (default)")
    args = ap.parse_args()

    do_tiny = not args.full
    if args.tiny:
        do_tiny = True

    cmd = ["python", "make_data.py", "--config", args.config, "--out", args.out]
    if do_tiny:
        cmd.append("--tiny")

    repo_root = os.path.dirname(os.path.abspath(__file__)) + "/.."
    run(cmd, cwd=repo_root)


if __name__ == "__main__":
    main()


