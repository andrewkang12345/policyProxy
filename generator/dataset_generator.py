from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any
import numpy as np
from .arenas import Arena, ArenaConfig
from .world import World, TeamConfig
from .lag import LagProcess, LagProcessConfig
from .utils import unit
from .policies import PolicyManager
from .schedulers import MixtureConfig, build_scheduler
from .opponent_policies import OpponentPolicyManager


@dataclass
class GeneratorConfig:
    # World
    seed: int = 0
    dt: float = 0.25
    steps: int = 80
    window: int = 6  # W
    teams: int = 2
    agents_per_team: int = 3
    selected_team: int = 0

    # Arena
    arena: ArenaConfig = None

    # Policies (pluggable)
    policies: List[Dict[str, Any]] | None = None  # list of PolicyConfig-like dicts
    mixture: Dict[str, Any] | None = None  # MixtureConfig-like dict
    # Optional: categories to auto-sample ego policies when "policies" is not provided
    policy_categories: List[str] | None = None  # e.g., ["discretized_deterministic", "smoothed_stochastic"]

    # Lag
    lag: LagProcessConfig = None

    # Opponent behavior
    opponent_policies: List[Dict[str, Any]] | None = None  # library for opponent steering
    opponent_mixture: Dict[str, Any] | None = None  # mixture config for opponents

    def __post_init__(self):
        if self.arena is None:
            self.arena = ArenaConfig()
        if self.policies is None:
            # Option A: sample from a single category (reproducible via seed)
            rng = np.random.default_rng(self.seed + 12345)
            # Determine category: if provided, take the FIRST only; else pick one
            if self.policy_categories and len(self.policy_categories) > 0:
                cat = str(self.policy_categories[0])
            else:
                cat = rng.choice([
                    "discretized_deterministic",
                    "discretized_stochastic",
                    "smoothed_deterministic",
                    "smoothed_stochastic",
                    "structured_flow_stochastic",
                ]).item()
            # Generate two distinct random policies within the SAME category
            out: List[Dict[str, Any]] = []
            for i in range(2):
                pid = f"P{i}"
                def attach_proto(meta: Dict[str, Any], style: str, stochastic: bool) -> Dict[str, Any]:
                    prot = dict(meta.get("prototypes", {}))
                    prot.setdefault("style", style)
                    if stochastic:
                        prot.setdefault("components", 4)
                    meta["prototypes"] = prot
                    return meta
                if cat == "discretized_deterministic":
                    bins = [int(rng.integers(8, 16)), int(rng.integers(8, 16))]
                    pol = {
                        "id": pid,
                        "family": "proto_actions",
                        "stochastic": False,
                        "regionizer": {"type": "discretized", "bins_per_dim": bins},
                    }
                    out.append(attach_proto(pol, "random", False))
                elif cat == "smoothed_stochastic":
                    ls = float(rng.uniform(0.6, 1.2))
                    sigma = float(rng.uniform(0.05, 0.2))
                    pol = {
                        "id": pid,
                        "family": "proto_actions",
                        "stochastic": True,
                        "noise": {"type": "gaussian", "sigma": sigma},
                        "regionizer": {"type": "smoothed", "smoother": "rbf", "length_scale": ls},
                    }
                    out.append(attach_proto(pol, "random", True))
                elif cat == "discretized_stochastic":
                    bins = [int(rng.integers(8, 16)), int(rng.integers(8, 16))]
                    sigma = float(rng.uniform(0.05, 0.2))
                    pol = {
                        "id": pid,
                        "family": "proto_actions",
                        "stochastic": True,
                        "noise": {"type": "gaussian", "sigma": sigma},
                        "regionizer": {"type": "discretized", "bins_per_dim": bins},
                    }
                    out.append(attach_proto(pol, "random", True))
                elif cat == "smoothed_deterministic":
                    ls = float(rng.uniform(0.6, 1.2))
                    pol = {
                        "id": pid,
                        "family": "proto_actions",
                        "stochastic": False,
                        "regionizer": {"type": "smoothed", "smoother": "rbf", "length_scale": ls},
                    }
                    out.append(attach_proto(pol, "random", False))
                elif cat == "structured_flow_stochastic":
                    bins = [int(rng.integers(10, 16)), int(rng.integers(10, 16))]
                    sigma = float(rng.uniform(0.02, 0.08))
                    pol = {
                        "id": pid,
                        "family": "proto_actions",
                        "stochastic": True,
                        "noise": {"type": "gaussian", "sigma": sigma},
                        "regionizer": {"type": "discretized", "bins_per_dim": bins},
                    }
                    out.append(attach_proto(pol, "structured", True))
                else:
                    # Fallback deterministic discretized
                    pol = {
                        "id": pid,
                        "family": "proto_actions",
                        "stochastic": False,
                        "regionizer": {"type": "discretized", "bins_per_dim": [12, 12]},
                    }
                    out.append(attach_proto(pol, "random", False))
            self.policies = out
        if self.mixture is None:
            self.mixture = {"scheduler": "markov_switch", "min_dwell_steps": 8}
        if self.lag is None:
            self.lag = LagProcessConfig()
        if self.opponent_policies is None:
            # Default: two random-mapping opponent policies using the state window over all teams
            self.opponent_policies = [
                {"id": "op0", "family": "random", "stochastic": True, "noise_sigma": 0.1, "regionizer": {"type": "window_hash", "length_scale": 0.25, "centers_per_dim": [2048, 1]}},
                {"id": "op1", "family": "random", "stochastic": True, "noise_sigma": 0.05, "regionizer": {"type": "window_hash", "length_scale": 0.2, "centers_per_dim": [2048, 1]}},
            ]
        if self.opponent_mixture is None:
            self.opponent_mixture = {"scheduler": "stagnant", "init_weights": [1.0] + [0.0] * (len(self.opponent_policies) - 1)}


class DatasetGenerator:
    def __init__(self, cfg: GeneratorConfig):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)

    def _init_world(self):
        arena = Arena(self.cfg.arena)
        team_cfgs = [TeamConfig(agent_count=self.cfg.agents_per_team) for _ in range(self.cfg.teams)]
        world = World(arena, team_cfgs, dt=self.cfg.dt, seed=self.rng.integers(1_000_000))
        return world

    def _opponent_actions(self, opp_man: OpponentPolicyManager, opp_sched, world_snap, team_idx: int, t: int, selected_team: int) -> np.ndarray:
        z = opp_sched[team_idx]
        pol = opp_man.policies[z]
        return pol.step(world_snap, selected_team, team_idx, t)

    # Deprecated intent-based helpers have been removed in favor of pluggable policies

    def generate_episode(self) -> dict:
        c = self.cfg
        world = self._init_world()
        lag_proc = LagProcess(c.lag, self.rng)
        # Policy manager and mixture scheduler for selected team
        arena = world.arena
        polman = PolicyManager(c.policies, agents=world.pos[c.selected_team].shape[0], arena_size=(arena.width, arena.height), seed=int(self.rng.integers(1_000_000)))
        mix_cfg = MixtureConfig(**(c.mixture or {}))
        sched = build_scheduler(mix_cfg, num_policies=len(polman.policies), seed=int(self.rng.integers(1_000_000)))
        z = sched.initial()

        # Opponent policy manager and per-team schedulers (excluding selected team)
        opp_man = OpponentPolicyManager(c.opponent_policies, agents=world.pos[1 - c.selected_team].shape[0], arena_size=(arena.width, arena.height), seed=int(self.rng.integers(1_000_000)))
        opp_mix_cfg = MixtureConfig(**(c.opponent_mixture or {}))
        # Build one scheduler per non-selected team (supports multi-opponent teams if extended)
        opp_sched = {}
        for team_i in range(c.teams):
            if team_i == c.selected_team:
                continue
            opp_sched[team_i] = build_scheduler(opp_mix_cfg, num_policies=len(opp_man.policies), seed=int(self.rng.integers(1_000_000)))
            # initialize state
            opp_sched[team_i]._z = opp_sched[team_i].initial()

        # First pass to create baseline snapshots for lag context references
        snapshots: List[dict] = []
        snap0 = world.step([np.zeros_like(p) for p in world.pos])
        snapshots.append(snap0)

        # Simulate with lag and policy mixture
        policy_ids: List[int] = []
        lags: List[int] = []
        snapshots = []
        world = self._init_world()
        opp_policy_ids: List[int] = []  # track per-step id of first opponent team (extendable)
        for t in range(c.steps):
            # Role-based logic removed

            # Build context for lag process
            if snapshots:
                pos_a = np.array(snapshots[-1]["pos"][0])
                pos_b = np.array(snapshots[-1]["pos"][1]) if len(snapshots[-1]["pos"]) > 1 else np.zeros((0, 2))
                if pos_b.shape[0] > 0:
                    d = np.linalg.norm(pos_a[:, None, :] - pos_b[None, :, :], axis=-1)
                    min_d = float(d.min())
                else:
                    min_d = 1e9
                context_state = {"min_interteam_dist": min_d}
            else:
                context_state = None

            lag_val = lag_proc.sample(context_state=context_state, agent_count=world.pos[c.selected_team].shape[0])
            if isinstance(lag_val, np.ndarray):
                ell = int(lag_val.max())
            else:
                ell = int(lag_val)
            lags.append(ell)

            if t > 0:
                z = sched.step(z, t)
                # advance opponent schedulers
                for team_i, sch in opp_sched.items():
                    sch._z = sch.step(sch._z, t)
            policy_ids.append(z)

            # Base lagged context for the environment lag
            t_ctx_base = max(0, t - ell)
            def get_snap_at(tt: int):
                if tt < len(snapshots):
                    return snapshots[tt]
                elif snapshots:
                    return snapshots[-1]
                else:
                    s0 = world.step([np.zeros_like(p) for p in world.pos])
                    snapshots.append(s0)
                    return s0

            actions = []
            for team_i in range(c.teams):
                if team_i == c.selected_team:
                    pol = polman.policies[z]
                    # Optional per-policy lag override
                    if getattr(pol.cfg, "use_lagged_state", True):
                        lag_extra = int(getattr(pol.cfg, "extra_lag_k", 0))
                        t_eff = max(0, t_ctx_base - lag_extra)
                        snap_eff = get_snap_at(t_eff)
                    else:
                        # Use most recent snapshot (approximate current state)
                        snap_eff = get_snap_at(t)
                    # Build windowed positions for regionizers that use temporal context
                    wstart = max(0, len(snapshots) - c.window + 1)
                    hist = snapshots[wstart:len(snapshots)] if snapshots else []
                    win_seq = [h["pos"] for h in hist] + [snap_eff["pos"]]
                    if len(win_seq) < c.window:
                        pad = [win_seq[0]] * (c.window - len(win_seq))
                        win_seq = pad + win_seq
                    win_pos = np.array(win_seq)
                    snap_ext = {**snap_eff, "win_pos": win_pos}
                    v = pol.step(snap_ext, c.selected_team, t)
                    actions.append(v)
                else:
                    # Use window-extended snapshot for opponents as well
                    snap_now = get_snap_at(t)
                    wstart_o = max(0, len(snapshots) - c.window + 1)
                    hist_o = snapshots[wstart_o:len(snapshots)] if snapshots else []
                    win_seq_o = [h["pos"] for h in hist_o] + [snap_now["pos"]]
                    if len(win_seq_o) < c.window:
                        pad_o = [win_seq_o[0]] * (c.window - len(win_seq_o))
                        win_seq_o = pad_o + win_seq_o
                    win_pos_o = np.array(win_seq_o)
                    snap_ext_o = {**snap_now, "win_pos": win_pos_o}
                    v = self._opponent_actions(opp_man, {team_i: opp_sched[team_i]._z}, snap_ext_o, team_i, t, c.selected_team)
                    actions.append(v)

            snap = world.step(actions)
            snapshots.append(snap)
            # Log opponent id (first non-selected team only)
            first_opp = 1 - c.selected_team if c.teams > 1 else None
            if first_opp is not None:
                opp_policy_ids.append(int(opp_sched[first_opp]._z))

        # Pack episode arrays
        pos_seq = [s["pos"] for s in snapshots]
        vel_seq = [s["vel"] for s in snapshots]

        T = len(pos_seq)
        team_count = len(pos_seq[0])
        agent_count = pos_seq[0][0].shape[0]
        pos = np.zeros((T, team_count, agent_count, 2))
        vel = np.zeros_like(pos)
        for t in range(T):
            for i in range(team_count):
                pos[t, i] = pos_seq[t][i]
                vel[t, i] = vel_seq[t][i]

        episode = {
            "pos": pos,
            "vel": vel,
            "lags": np.array(lags, dtype=int),
            "policy_ids": np.array(policy_ids, dtype=int),
            "opp_policy_ids": np.array(opp_policy_ids, dtype=int) if len(opp_policy_ids) > 0 else None,
            "meta": {
                "policies": polman.describe(),
                "mixture": sched.describe(),
                "opponent_policies": opp_man.describe(),
                "opponent_mixture": opp_mix_cfg.__dict__,
                "selected_team": self.cfg.selected_team,
                "window": self.cfg.window,
                "dt": self.cfg.dt,
            },
        }
        return episode

