from __future__ import annotations
import argparse
import os
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_json(path: str):
    with open(path, "r") as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser(description="Consolidated CVAE robustness report (PDF)")
    ap.add_argument("--eval_json", type=str, required=True)
    ap.add_argument("--repr_json", type=str, required=True)
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()

    evalj = load_json(args.eval_json)
    reprj = load_json(args.repr_json)

    splits = sorted(evalj["splits"].keys())
    # Build metrics arrays
    ade = [evalj["splits"][s]["metrics"]["ADE"] for s in splits]
    fde = [evalj["splits"][s]["metrics"]["FDE"] for s in splits]
    js_state = [evalj["splits"][s]["divergences"]["js_state"] for s in splits]
    js_action = [evalj["splits"][s]["divergences"]["js_action"] for s in splits]
    js_policy = [evalj["splits"][s]["divergences"]["js_policy"] for s in splits]

    probe_splits = sorted(reprj.keys())
    probe_acc = [reprj[s]["probe_accuracy"] for s in probe_splits]

    fig, axes = plt.subplots(3, 2, figsize=(14, 12))

    # Divergences
    ax = axes[0, 0]
    x = range(len(splits))
    ax.plot(x, js_state, marker="o", label="JS(state)")
    ax.plot(x, js_action, marker="o", label="JS(action)")
    ax.plot(x, js_policy, marker="o", label="JS(policy)")
    ax.set_title("Divergences vs Train")
    ax.set_xticks(x)
    ax.set_xticklabels(splits, rotation=30, ha="right")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # ADE/FDE
    ax = axes[1, 0]
    w = 0.35
    x = range(len(splits))
    ax.bar([i - w/2 for i in x], ade, width=w, label="ADE")
    ax.bar([i + w/2 for i in x], fde, width=w, label="FDE")
    ax.set_title("Prediction Errors (lower is better)")
    ax.set_xticks(x)
    ax.set_xticklabels(splits, rotation=30, ha="right")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Probe accuracy
    ax = axes[2, 0]
    x = range(len(probe_splits))
    ax.bar(x, probe_acc, color="#4c78a8")
    ax.set_title("Policy Representation Probe Accuracy")
    ax.set_xticks(x)
    ax.set_xticklabels(probe_splits, rotation=30, ha="right")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.3)

    # Scatter: divergence vs ADE for state/action/policy
    def scatter_div_err(ax, divs, errs, title):
        ax.scatter(divs, errs, c="#e45756")
        ax.set_title(title)
        ax.set_xlabel("JS divergence")
        ax.set_ylabel("ADE")
        ax.grid(True, alpha=0.3)

    # Prepare arrays excluding 'train' since it is 0 divergence by design
    spl_nontrain = [s for s in splits if s != "train"]
    ade_nt = [evalj["splits"][s]["metrics"]["ADE"] for s in spl_nontrain]
    s_nt = [evalj["splits"][s]["divergences"]["js_state"] for s in spl_nontrain]
    a_nt = [evalj["splits"][s]["divergences"]["js_action"] for s in spl_nontrain]
    p_nt = [evalj["splits"][s]["divergences"]["js_policy"] for s in spl_nontrain]

    scatter_div_err(axes[0, 1], s_nt, ade_nt, "State divergence vs ADE")
    scatter_div_err(axes[1, 1], a_nt, ade_nt, "Action divergence vs ADE")
    scatter_div_err(axes[2, 1], p_nt, ade_nt, "Policy divergence vs ADE")

    plt.tight_layout()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
