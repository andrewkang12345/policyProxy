from __future__ import annotations
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

from tasks.common.dataset import load_index, load_episode


def _load_episode_from_args(episode_path: str | None, index_path: str | None, ep_idx: int | None):
    if episode_path:
        return load_episode(episode_path)
    if index_path is None or ep_idx is None:
        raise ValueError("Provide either --episode or both --index and --ep")
    index = load_index(index_path)
    if not (0 <= ep_idx < len(index)):
        raise IndexError(f"ep index {ep_idx} out of range [0, {len(index)-1}]")
    return load_episode(index[ep_idx]["path"])


def animate_episode(ep: dict, out_path: str, fps: int = 8, dpi: int = 120, tail: int = 0, selected_team: int | None = None):
    pos: np.ndarray = ep["pos"]  # [T, teams, agents, 2]
    policy_ids: np.ndarray = ep.get("intents", ep.get("policy_ids", np.zeros(pos.shape[0], dtype=int)))  # [T]
    T, teams, agents, _ = pos.shape
    meta = ep.get("meta", {})
    arena = meta.get("arena", {"width": 20.0, "height": 14.0})
    width = float(arena.get("width", 20.0))
    height = float(arena.get("height", 14.0))
    if selected_team is None:
        selected_team = int(meta.get("selected_team", 0))
    
    # Get policy names if available
    policies = meta.get("policies", [])
    policy_names = {i: p.get("id", f"P{i}") for i, p in enumerate(policies)} if policies else {}

    # Setup figure
    fig, ax = plt.subplots(figsize=(width / 2.0, height / 2.0))
    ax.set_xlim(0.0, width)
    ax.set_ylim(0.0, height)
    ax.set_aspect('equal', adjustable='box')
    ax.set_title("Episode Visualization")
    ax.grid(True, alpha=0.2)
    
    # Add text for policy display
    policy_text = ax.text(0.02, 0.98, "", transform=ax.transAxes, fontsize=12, 
                         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    colors = []
    for i in range(teams):
        if i == selected_team:
            colors.append('#1f77b4')  # blue for ego
        else:
            colors.append('#e45756')  # red for opponents

    scatters = []
    trails = []
    for i in range(teams):
        sc = ax.scatter([], [], s=60, c=colors[i], edgecolors='k', linewidths=0.5, label=("ego" if i == selected_team else f"team {i}"))
        scatters.append(sc)
        if tail > 0:
            # Per-agent trails instead of concatenated team trails
            team_trails = []
            for a in range(agents):
                ln, = ax.plot([], [], '-', color=colors[i], alpha=0.3, linewidth=1.0)
                team_trails.append(ln)
            trails.append(team_trails)
        else:
            trails.append(None)
    ax.legend(loc='upper right')

    def init():
        for i in range(teams):
            scatters[i].set_offsets(np.zeros((agents, 2)))
            if trails[i] is not None:
                for agent_trail in trails[i]:
                    agent_trail.set_data([], [])
        policy_text.set_text("")
        all_trails = [t for team_trails in trails if team_trails is not None for t in team_trails]
        return scatters + all_trails + [policy_text]

    def update(frame: int):
        for i in range(teams):
            pts = pos[frame, i]
            scatters[i].set_offsets(pts)
            if trails[i] is not None and frame > 0:
                t0 = max(0, frame - tail)
                # Per-agent trails
                for a in range(agents):
                    xs = pos[t0:frame+1, i, a, 0]
                    ys = pos[t0:frame+1, i, a, 1]
                    trails[i][a].set_data(xs, ys)
        
        # Update policy text
        if frame < len(policy_ids):
            pid = int(policy_ids[frame])
            policy_name = policy_names.get(pid, f"Policy {pid}")
            policy_text.set_text(f"Frame {frame}/{T-1}\nEgo Policy: {policy_name}")
        
        all_trails = [t for team_trails in trails if team_trails is not None for t in team_trails]
        return scatters + all_trails + [policy_text]

    interval_ms = int(1000 / max(1, fps))
    anim = animation.FuncAnimation(fig, update, init_func=init, frames=T, interval=interval_ms, blit=True)

    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    anim.save(out_path, writer='pillow', dpi=dpi, savefig_kwargs={'facecolor': 'white'})
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Visualize a single episode to GIF")
    ap.add_argument("--episode", type=str, default=None, help="Path to episode .npz")
    ap.add_argument("--index", type=str, default=None, help="Path to index.json (use with --ep)")
    ap.add_argument("--ep", type=int, default=None, help="Episode index within index.json")
    ap.add_argument("--out", type=str, required=True, help="Output GIF path")
    ap.add_argument("--fps", type=int, default=8)
    ap.add_argument("--dpi", type=int, default=120)
    ap.add_argument("--tail", type=int, default=0, help="Show trail of last N frames (0 to disable)")
    ap.add_argument("--selected_team", type=int, default=None)
    args = ap.parse_args()

    ep = _load_episode_from_args(args.episode, args.index, args.ep)
    animate_episode(ep, out_path=args.out, fps=args.fps, dpi=args.dpi, tail=max(0, int(args.tail)), selected_team=args.selected_team)


if __name__ == "__main__":
    main()


