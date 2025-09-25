from __future__ import annotations
import argparse
import os
import json
import matplotlib
import matplotlib.pyplot as plt
from dataclasses import dataclass
import numpy as np
import yaml
import torch
from torch import nn

from generator.arenas import Arena, ArenaConfig
from generator.world import World, TeamConfig
from generator.policies import PolicyManager
from generator.schedulers import MixtureConfig, build_scheduler
from tasks.common.dataset import NextFrameDataset


def load_config_used(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, 1e-8)


from generator.opponent_policies import OpponentPolicyManager, OpponentPolicyConfig


class GRUModel(nn.Module):
    def __init__(self, teams: int, agents: int, hidden: int = 128):
        super().__init__()
        self.teams = teams
        self.agents = agents
        self.in_dim = teams * agents * 2
        self.gru = nn.GRU(self.in_dim, hidden, batch_first=True)
        self.head = nn.Linear(hidden, agents * 2)
    def forward(self, x):
        B, W, T, A, D = x.shape
        x = x.reshape(B, W, T * A * D)
        h, _ = self.gru(x)
        y = self.head(h[:, -1, :])
        return y.reshape(B, self.agents, 2)


class TemporalAttentionModel(nn.Module):
    def __init__(self, teams: int, agents: int, window: int, hidden: int = 128):
        super().__init__()
        self.teams = teams
        self.agents = agents
        self.window = window
        self.in_dim = teams * agents * 2
        self.key = nn.Linear(self.in_dim, hidden)
        self.query = nn.Linear(self.in_dim, hidden)
        self.value = nn.Linear(self.in_dim, hidden)
        self.out = nn.Sequential(nn.ReLU(), nn.Linear(hidden, agents * 2))
    def forward(self, x):
        B, W, T, A, D = x.shape
        x = x.reshape(B, W, T * A * D)
        K = self.key(x)
        Q = self.query(x[:, -1, :]).unsqueeze(1)
        V = self.value(x)
        attn = torch.softmax((Q @ K.transpose(1, 2)) / (K.size(-1) ** 0.5), dim=-1)
        ctx = attn @ V
        y = self.out(ctx.squeeze(1))
        return y.reshape(B, self.agents, 2), attn.squeeze(1)


@dataclass
class RolloutConfig:
    segmented_z: bool = False


class RolloutModelWrapper:
    def __init__(self, state_dict: dict, teams: int, agents: int, window: int, device: str, segmented_z: bool):
        self.device = device
        self.segmented_z = segmented_z
        self.latent_cache: torch.Tensor | None = None
        self.model: nn.Module | None = None
        self.is_cvae = False
        self.requires_policy_id = False
        self.latent_dim: int | None = None
        self.policy_latent_bank: dict[int | str, torch.Tensor] = {}

        if "state_dict" in state_dict and isinstance(state_dict["state_dict"], dict):
            state_dict = state_dict["state_dict"]
        if isinstance(state_dict, list):
            raise RuntimeError("Unexpected checkpoint format for rollout")

        latent_dim = None
        hidden_dim = 128
        for k, v in state_dict.items():
            if k.endswith("to_mu.weight") and latent_dim is None:
                latent_dim = v.shape[0]
            if k.startswith("enc.gru.weight_hh_l0"):
                hidden_dim = v.shape[1]
        self.latent_dim = latent_dim

        policy_embed = state_dict.get("policy_embed.weight")
        try:
            from baselines.state_cond.train_cvae_pid import PolicyCVAE
        except Exception:
            PolicyCVAE = None
        try:
            from baselines.state_cond.train_cvae import CVAE
        except Exception:
            CVAE = None

        if PolicyCVAE and policy_embed is not None and latent_dim is not None:
            num_policies = policy_embed.shape[0]
            embed_dim = policy_embed.shape[1]
            model = PolicyCVAE(
                teams=teams,
                agents=agents,
                window=window,
                num_policies=num_policies,
                hidden=hidden_dim,
                latent=latent_dim,
                embed_dim=embed_dim,
            )
            model.load_state_dict(state_dict, strict=False)
            self.model = model.to(device)
            self.model.eval()
            self.is_cvae = True
            self.requires_policy_id = True
            return

        if CVAE and latent_dim is not None:
            global_z = "z_global" in state_dict
            model = CVAE(teams=teams, agents=agents, window=window, hidden=hidden_dim, latent=latent_dim, global_z=global_z)
            model.load_state_dict(state_dict, strict=False)
            self.model = model.to(device)
            self.model.eval()
            self.is_cvae = True
            return

        model = GRUModel(teams=teams, agents=agents)
        try:
            model.load_state_dict(state_dict, strict=False)
        except Exception:
            model = TemporalAttentionModel(teams=teams, agents=agents, window=window)
            model.load_state_dict(state_dict, strict=False)
        self.model = model.to(device)
        self.model.eval()

    def reset(self):
        self.latent_cache = None

    def _sample_latent(self) -> torch.Tensor | None:
        if not self.is_cvae or self.latent_dim is None:
            return None
        if hasattr(self.model, "z_global"):
            return self.model.z_global.unsqueeze(0)
        if self.segmented_z:
            if self.latent_cache is None:
                self.latent_cache = torch.randn(1, self.latent_dim, device=self.device)
            return self.latent_cache
        return torch.randn(1, self.latent_dim, device=self.device)

    def predict(self, window_tensor: torch.Tensor, policy_id: torch.Tensor | None = None) -> torch.Tensor:
        if not self.is_cvae:
            out = self.model(window_tensor)
            if isinstance(out, tuple):
                out = out[0]
            return out

        z = self._sample_latent()
        if hasattr(self.model, "policy_embed"):
            if policy_id is None:
                raise RuntimeError("Policy-conditioned CVAE requires policy_id inputs during rollout")
            key = int(policy_id.item())
            if key in self.policy_latent_bank:
                z = self.policy_latent_bank[key].unsqueeze(0)
            elif z is None:
                z = torch.zeros((1, self.latent_dim), device=self.device)
            out = self.model.decode(window_tensor, z, policy_id)
            return out[..., :2]
        if z is None:
            raise RuntimeError("Unable to sample latent for CVAE rollout")
        out = self.model.decode(window_tensor, z)
        return out[..., :2]

    def build_latent_bank(self, data_root: str, max_items: int = 2048) -> None:
        if not self.is_cvae:
            return
        index_path = os.path.join(data_root, "train", "index.json")
        if not os.path.exists(index_path):
            return
        dataset = NextFrameDataset(index_path, label_kind="policy")
        counts: dict[int | str, int] = {}
        sums: dict[int | str, torch.Tensor] = {}
        total = 0
        for item in dataset:
            s = torch.tensor(item["state"][None, ...], dtype=torch.float32, device=self.device)
            a = torch.tensor(item["action"][None, ...], dtype=torch.float32, device=self.device)
            pid = int(item["policy_id"])
            with torch.no_grad():
                if self.requires_policy_id:
                    p = torch.tensor([pid], dtype=torch.long, device=self.device)
                    mu, _ = self.model.encode(s, a, p)
                else:
                    mu, _ = self.model.encode(s, a)
            key: int | str = pid if self.requires_policy_id else "default"
            mu = mu.squeeze(0).detach()
            if key not in sums:
                sums[key] = torch.zeros_like(mu)
            sums[key] += mu
            counts[key] = counts.get(key, 0) + 1
            total += 1
            if total >= max_items:
                break
        for key, value in sums.items():
            avg = value / max(1, counts[key])
            self.policy_latent_bank[key] = avg.to(self.device)


def compute_metrics(traj_pos: list[np.ndarray], traj_vel: list[np.ndarray], selected_team: int) -> dict:
    # traj_pos: list over time of [teams][agents,2]
    # traj_vel: list over time of [teams][agents,2]
    T = len(traj_pos)
    # Simple collision: any inter-team pair within 0.5
    collisions = 0
    checks = 0
    for t in range(T):
        a = traj_pos[t][selected_team]
        b = traj_pos[t][1 - selected_team]
        da = a[:, None, :] - b[None, :, :]
        d = np.linalg.norm(da, axis=-1)
        collisions += int((d < 0.5).sum())
        checks += d.size
    collision_rate = float(collisions / max(1, checks))
    # Smoothness: mean jerk on selected team
    vel = np.array([v[selected_team] for v in traj_vel])
    accel = np.diff(vel, axis=0)
    jerk = np.diff(accel, axis=0)
    smooth = float(np.mean(np.linalg.norm(jerk, axis=-1))) if jerk.size else 0.0
    return {
        "collision_rate": collision_rate,
        "smoothness": smooth,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, default="data/v1.0")
    ap.add_argument("--model", type=str, required=True)
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--save_json", type=str, default=None, help="Path to save metrics JSON (per-episode and aggregate)")
    ap.add_argument("--save_plot", type=str, default=None, help="Path to save summary plot (PNG/PDF)")
    ap.add_argument("--segmented_z", action="store_true", help="Use segmented latent per rollout for CVAE models")
    args = ap.parse_args()

    # Normalize device aliases
    if str(args.device).lower() == "gpu":
        args.device = "cuda"

    cfg_path = os.path.join(args.data_root, "config_used.yaml")
    cfg = load_config_used(cfg_path)
    gcfg = cfg["generator"]

    arena = Arena(ArenaConfig(**gcfg["arena"]))
    teams = int(gcfg["teams"])
    agents = int(gcfg["agents_per_team"])
    selected_team = int(gcfg["selected_team"])
    dt = float(gcfg["dt"])
    steps = int(gcfg["steps"])
    window = int(gcfg["window"])
    # Build opponent manager from config used
    opp_cfgs = gcfg.get("opponent_policies")
    if not opp_cfgs:
        opp_cfgs = [
            {"id": "guard", "variant": "guard", "stochastic": True, "noise_sigma": 0.1},
            {"id": "pursue", "variant": "pursue", "stochastic": False},
        ]

    # Load policy mixture metadata if available
    mixture_cfg = gcfg.get("mixture")
    policies_meta = gcfg.get("policies")

    team_cfgs = [TeamConfig(agent_count=agents) for _ in range(teams)]
    dummy_world = World(arena, team_cfgs, dt=dt)
    Tteams = len(dummy_world.pos)
    Aagents = dummy_world.pos[0].shape[0]

    state = torch.load(args.model, map_location=args.device)
    wrapper = RolloutModelWrapper(state, teams=Tteams, agents=Aagents, window=window, device=args.device, segmented_z=args.segmented_z)
    if args.segmented_z:
        wrapper.build_latent_bank(args.data_root)

    rng = np.random.default_rng(123)
    opp_man = OpponentPolicyManager(opp_cfgs, agents=agents, arena_size=(arena.width, arena.height), seed=1234)

    policy_manager = None
    scheduler = None
    if policies_meta and mixture_cfg:
        policy_manager = PolicyManager(policies_meta, agents=agents, arena_size=(arena.width, arena.height), seed=cfg["generator"].get("seed", 0) + 100)
        mix_cfg = MixtureConfig(**mixture_cfg)
        scheduler = build_scheduler(mix_cfg, num_policies=len(policy_manager.policies), seed=cfg["generator"].get("seed", 0) + 200)

    results = []
    for ep in range(args.episodes):
        world = World(arena, [TeamConfig(agent_count=agents) for _ in range(teams)], dt=dt, seed=int(rng.integers(1_000_000)))
        wrapper.reset()
        # Prime with zeros to fill initial window
        traj_pos = []
        traj_vel = []
        snap = world.step([np.zeros_like(p) for p in world.pos])
        traj_pos.append([p.copy() for p in snap["pos"]])
        traj_vel.append([v.copy() for v in snap["vel"]])
        history = [snap["pos"]]
        current_policy_id = 0
        if scheduler is not None:
            current_policy_id = scheduler.initial()
        for t in range(1, steps):
            if scheduler is not None:
                current_policy_id = scheduler.step(current_policy_id, t)
            # Build input window
            win = np.array(history[-window:])
            if win.shape[0] < window:
                pad = [history[0]] * (window - win.shape[0])
                win = np.array(pad + history)
            s_t = torch.tensor(win[None, ...], dtype=torch.float32, device=args.device)
            policy_tensor = None
            if wrapper.requires_policy_id:
                policy_tensor = torch.tensor([current_policy_id], dtype=torch.long, device=args.device)
            with torch.no_grad():
                pred = wrapper.predict(s_t, policy_tensor)
                if isinstance(pred, tuple):
                    pred = pred[0]
                a_sel = pred[0].cpu().numpy()
            acts = []
            for team_idx in range(teams):
                if team_idx == selected_team:
                    acts.append(a_sel)
                else:
                    pol = opp_man.policies[0]
                    acts.append(pol.step({"pos": history[-1]}, selected_team, team_idx, t))
            snap = world.step(acts)
            traj_pos.append([p.copy() for p in snap["pos"]])
            traj_vel.append([v.copy() for v in snap["vel"]])
            history.append(snap["pos"])
        metrics = compute_metrics(traj_pos, traj_vel, selected_team)
        results.append(metrics)

    agg = {k: float(np.mean([r[k] for r in results])) for k in results[0].keys()}
    payload = {"per_episode": results, "aggregate": agg}
    print(json.dumps(payload, indent=2))

    if args.save_json:
        os.makedirs(os.path.dirname(args.save_json), exist_ok=True)
        with open(args.save_json, "w") as f:
            json.dump(payload, f, indent=2)

    if args.save_plot:
        try:
            plt.switch_backend("Agg")
        except Exception:
            pass
        metrics = list(agg.keys())
        values = [agg[m] for m in metrics]
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        ax0, ax1 = axes
        ax0.bar(metrics, values, color=["#4c78a8", "#f58518", "#e45756", "#72b7b2"][: len(metrics)])
        ax0.set_title("Rollout aggregate metrics")
        ax0.set_ylabel("Value")
        ax0.set_xticklabels(metrics, rotation=30, ha="right")
        for m_idx, m in enumerate(metrics):
            y = [r[m] for r in results]
            x = np.full(len(y), m_idx) + (np.arange(len(y)) - (len(y) - 1) / 2.0) * (0.08 / max(1, len(y) - 1))
            ax1.scatter(x, y, label=m)
        ax1.set_xticks(range(len(metrics)))
        ax1.set_xticklabels(metrics, rotation=30, ha="right")
        ax1.set_title("Per-episode metrics")
        ax1.grid(True, alpha=0.3)
        plt.tight_layout()
        os.makedirs(os.path.dirname(args.save_plot), exist_ok=True)
        fig.savefig(args.save_plot, dpi=150, bbox_inches="tight")
        plt.close(fig)


if __name__ == "__main__":
    main()



