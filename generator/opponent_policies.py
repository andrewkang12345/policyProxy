from __future__ import annotations
from dataclasses import dataclass, replace
from typing import Dict, Any, List, Tuple
import numpy as np

from .regionizers import RegionizerConfig, build_regionizer, DiscretizedRegionizer, SmoothedRegionizer, WindowHashRegionizer

@dataclass
class OpponentPolicyConfig:
    id: str
    family: str = "random"
    stochastic: bool = True
    noise_sigma: float = 0.1
    # Optional:
    bias: str | None = None  # 'state_shift' | 'action_shift' | None
    severity: float = 0.0    # 0.0 (off) .. 1.0 (strong)
    # For random mapping opponents
    regionizer: Dict[str, Any] | None = None
    # Movement-based opponent (no state conditioning)
    variant: str | None = None  # 'movement' to use movement opponent
    movement_dir: float | None = None  # radians; if None, random per-episode
    movement_speed: float | None = None  # scalar speed multiplier
    # Movement NN variant (no state conditioning)
    nn_hidden: int | None = None
    nn_params: List[float] | None = None  # flattened parameters
    nn_scale: float | None = None
    nn_amplitude: float | None = None
    nn_phase: List[float] | None = None
    nn_mix: float | None = None


class OpponentPolicy:
    def __init__(self, cfg: OpponentPolicyConfig, agents: int, arena_size: Tuple[float, float], seed: int):
        self.cfg = cfg
        self.id = cfg.id
        self.agents = int(agents)
        self.arena = (float(arena_size[0]), float(arena_size[1]))
        self.rng = np.random.default_rng(seed)

    def step(self, snap: Dict[str, Any], selected_team: int, team_idx: int, t: int) -> np.ndarray:
        raise NotImplementedError

    def _noise(self, shape) -> np.ndarray:
        if not self.cfg.stochastic or self.cfg.noise_sigma <= 0:
            return 0.0
        return self.rng.normal(scale=self.cfg.noise_sigma, size=shape)


class RandomMappingOpponent(OpponentPolicy):
    def __init__(self, cfg: OpponentPolicyConfig, agents: int, arena_size: Tuple[float, float], seed: int):
        super().__init__(cfg, agents, arena_size, seed)
        rcfg = RegionizerConfig(**(cfg.regionizer or {"type": "window_hash", "length_scale": 0.25, "centers_per_dim": [2048, 1]}))
        self.regionizer = build_regionizer(rcfg)
        # Determine number of prototype bins K
        if isinstance(self.regionizer, DiscretizedRegionizer):
            K = self.regionizer.bins_x * self.regionizer.bins_y
        elif isinstance(self.regionizer, SmoothedRegionizer):
            K = self.regionizer.centers.shape[0]
        elif isinstance(self.regionizer, WindowHashRegionizer):
            K = int(getattr(self.regionizer, "num_buckets", 1024))
        else:
            K = 1024
        v = self.rng.normal(size=(K, self.agents, 2))
        n = np.linalg.norm(v, axis=-1, keepdims=True)
        self.prototypes = (v / np.maximum(n, 1e-6)).astype(float)

    def step(self, snap: Dict[str, Any], selected_team: int, team_idx: int, t: int) -> np.ndarray:
        # Build mean action from regionized state window (all teams)
        if "win_pos" in snap:
            win_pos = snap["win_pos"]
            if isinstance(self.regionizer, WindowHashRegionizer):
                idx = self.regionizer.bin_index_from_window(win_pos, selected_team, self.arena)
                mean = self.prototypes[idx % self.prototypes.shape[0]]
            else:
                feat = self.regionizer.state_feature_window(win_pos, selected_team, self.arena)
                if isinstance(self.regionizer, DiscretizedRegionizer):
                    _, _, idx = self.regionizer.bin_index(feat)
                    mean = self.prototypes[idx]
                elif isinstance(self.regionizer, SmoothedRegionizer):
                    phi = self.regionizer.rbf_phi(feat)
                    mean = (phi[:, None, None] * self.prototypes).sum(axis=0)
                else:
                    mean = self.prototypes[0]
        else:
            pos_all = snap["pos"]
            feat = self.regionizer.state_feature(pos_all, selected_team, self.arena)
            if isinstance(self.regionizer, DiscretizedRegionizer):
                _, _, idx = self.regionizer.bin_index(feat)
                mean = self.prototypes[idx]
            elif isinstance(self.regionizer, SmoothedRegionizer):
                phi = self.regionizer.rbf_phi(feat)
                mean = (phi[:, None, None] * self.prototypes).sum(axis=0)
            elif isinstance(self.regionizer, WindowHashRegionizer):
                # Fallback hash based on last frame only
                W = np.array(pos_all)
                W = W[None, ...]
                idx = self.regionizer.bin_index_from_window(W, selected_team, self.arena)
                mean = self.prototypes[idx % self.prototypes.shape[0]]
            else:
                mean = self.prototypes[0]

        # Optional bias to steer distributions
        if self.cfg.bias and self.cfg.severity > 0:
            sev = float(np.clip(self.cfg.severity, 0.0, 1.0))
            pos_all = snap["pos"]
            my = np.array(pos_all[team_idx])
            sel = np.array(pos_all[selected_team])
            my_cent = my.mean(axis=0)
            sel_cent = sel.mean(axis=0)
            if self.cfg.bias == "state_shift":
                push = (my - sel_cent)
                push /= np.maximum(np.linalg.norm(push, axis=-1, keepdims=True), 1e-6)
                mean = (1 - sev) * mean + sev * push
            elif self.cfg.bias == "action_shift":
                goal = np.array([self.arena[0] - 2.0, self.arena[1] / 2.0])
                carrier = sel[0]
                mid = (carrier + goal) / 2.0
                pull = mid - my
                pull /= np.maximum(np.linalg.norm(pull, axis=-1, keepdims=True), 1e-6)
                mean = (1 - sev) * mean + sev * pull
            nrm = np.linalg.norm(mean, axis=-1, keepdims=True)
            mean = mean / np.maximum(nrm, 1e-6)

        act = mean + self._noise(mean.shape)
        n2 = np.linalg.norm(act, axis=-1, keepdims=True)
        return act / np.maximum(n2, 1e-6)


class MovementOpponent(OpponentPolicy):
    """
    Movement-only opponent: produces actions as constant-direction velocity vectors 
    with configurable speed and per-step Gaussian noise. Ignores state entirely.
    """
    def __init__(self, cfg: OpponentPolicyConfig, agents: int, arena_size: Tuple[float, float], seed: int):
        super().__init__(cfg, agents, arena_size, seed)
        ang = cfg.movement_dir if cfg.movement_dir is not None else float(self.rng.uniform(-np.pi, np.pi))
        spd = cfg.movement_speed if cfg.movement_speed is not None else 1.0
        self.base_dir = np.array([np.cos(ang), np.sin(ang)], dtype=float)
        self.speed = float(spd)

    def step(self, snap: Dict[str, Any], selected_team: int, team_idx: int, t: int) -> np.ndarray:
        # Same velocity for all agents, optionally with noise
        mean = np.tile(self.base_dir[None, :], (self.agents, 1)) * self.speed
        act = mean + self._noise(mean.shape)
        n2 = np.linalg.norm(act, axis=-1, keepdims=True)
        return act / np.maximum(n2, 1e-6)


class MovementNNOpponent(OpponentPolicy):
    """
    Movement-NN opponent: ignores state; uses a tiny MLP over time index to generate
    per-agent velocity patterns. This provides a flexible parameterization to tune.
    """
    def __init__(self, cfg: OpponentPolicyConfig, agents: int, arena_size: Tuple[float, float], seed: int):
        super().__init__(cfg, agents, arena_size, seed)
        self.hidden = int(getattr(cfg, "nn_hidden", 32) or 32)
        self.scale = float(getattr(cfg, "nn_scale", 0.8) or 0.8)
        # Param shapes: W1:[1,self.hidden], b1:[self.hidden], W2:[self.hidden,2], b2:[2]
        # One shared MLP for all agents; add per-agent phase shifts
        total = 1 * self.hidden + self.hidden + self.hidden * 2 + 2
        params = np.array(getattr(cfg, "nn_params", []), dtype=float)
        if params.size != total:
            self.W1 = self.rng.normal(0, 0.2, size=(1, self.hidden))
            self.b1 = np.zeros(self.hidden)
            self.W2 = self.rng.normal(0, 0.2, size=(self.hidden, 2))
            self.b2 = np.zeros(2)
        else:
            off = 0
            self.W1 = params[off:off + 1 * self.hidden].reshape(1, self.hidden); off += 1 * self.hidden
            self.b1 = params[off:off + self.hidden]; off += self.hidden
            self.W2 = params[off:off + self.hidden * 2].reshape(self.hidden, 2); off += self.hidden * 2
            self.b2 = params[off:off + 2]
        # Per-agent phase for temporal input
        phase = getattr(cfg, "nn_phase", None)
        if phase is not None and len(phase) == agents:
            self.agent_phase = np.array(phase, dtype=float)
        else:
            self.agent_phase = np.linspace(0.0, 2 * np.pi, num=agents, endpoint=False)
        # Deterministic amplitude so tuning remains stable
        amp = getattr(cfg, "nn_amplitude", None)
        if amp is None:
            amp = 4.5  # Default amplitude roughly matching prior stochastic scaling
        self.movement_amplitude = float(max(0.05, amp))
        # Blend factor between baseline random mapping and movement NN
        self.mix = float(np.clip(getattr(cfg, "nn_mix", 1.0) or 1.0, 0.0, 1.0))

        # Optional baseline random opponent to support interpolation back toward original behavior
        baseline_cfg = replace(cfg, variant=None)
        # Ensure NN-specific fields don't interfere with baseline opponent
        baseline_cfg.nn_params = None
        baseline_cfg.nn_hidden = None
        baseline_cfg.nn_scale = None
        baseline_cfg.nn_amplitude = None
        baseline_cfg.nn_phase = None
        baseline_cfg.nn_mix = None
        self._baseline = RandomMappingOpponent(baseline_cfg, agents, arena_size, seed + 17)

    def _forward(self, t: int) -> np.ndarray:
        # Time embedding with sin/cos and per-agent phase
        tau = 2 * np.pi * (t / 100.0)
        phase = self.agent_phase
        # Shared MLP on time scalar, then tile per agent
        x = np.array([[np.sin(tau), np.cos(tau)]])  # shape [1,2]
        # Project to 1-dim first to fit W1 shape
        x1 = x @ np.array([[0.7],[0.3]])  # [1,1]
        h = np.tanh(x1 @ self.W1 + self.b1)
        out = np.tanh(h @ self.W2 + self.b2)  # [1,2]
        base = out.reshape(1, 2)
        # Add per-agent rotation via phase and amplitude
        ca = np.cos(phase)[:, None]
        sa = np.sin(phase)[:, None]
        rot = np.concatenate([np.stack([ca, -sa], axis=-1), np.stack([sa, ca], axis=-1)], axis=-2)  # [A,2,2]
        vec = np.tile(base, (self.agents, 1))  # [A,2]
        vec = (rot @ vec[..., None]).squeeze(-1)
        # Apply learned scale but keep vector normalized; amplitude handled in step
        vec = vec * self.scale
        nrm = np.linalg.norm(vec, axis=-1, keepdims=True)
        return vec / np.maximum(nrm, 1e-6)

    def step(self, snap: Dict[str, Any], selected_team: int, team_idx: int, t: int) -> np.ndarray:
        base = self._forward(t)
        mean = base * self.movement_amplitude
        if self.mix < 1.0:
            baseline = self._baseline.step(snap, selected_team, team_idx, t)
            mean = self.mix * mean + (1.0 - self.mix) * baseline
        act = mean + self._noise(mean.shape)
        nrm = np.linalg.norm(act, axis=-1, keepdims=True)
        if np.any(nrm > 1e-6):
            # Clamp to target speed while preserving deviations
            target = np.clip(self.movement_amplitude, 0.05, 3.0)
            act = act / np.maximum(nrm, 1e-6) * target
        return act

def build_opponent_policy(cfg: OpponentPolicyConfig, agents: int, arena_size: Tuple[float, float], seed: int) -> OpponentPolicy:
    if cfg.family == "random":
        # Movement-only variant: ignores state; produces fixed-direction velocity with noise
        if getattr(cfg, "variant", None) == "movement":
            return MovementOpponent(cfg, agents, arena_size, seed)
        if getattr(cfg, "variant", None) == "movement_nn":
            return MovementNNOpponent(cfg, agents, arena_size, seed)
        return RandomMappingOpponent(cfg, agents, arena_size, seed)
    else:
        raise ValueError(f"Unknown opponent policy family: {cfg.family}")


class OpponentPolicyManager:
    def __init__(self, policy_cfgs: List[Dict[str, Any]], agents: int, arena_size: Tuple[float, float], seed: int):
        self.policies: List[OpponentPolicy] = []
        self.id_to_index: Dict[str, int] = {}
        for i, pcfg in enumerate(policy_cfgs):
            cfg = OpponentPolicyConfig(**pcfg)
            pol = build_opponent_policy(cfg, agents, arena_size, seed + i * 131)
            self.policies.append(pol)
            self.id_to_index[cfg.id] = i

    def get_by_id(self, pid: str) -> OpponentPolicy:
        return self.policies[self.id_to_index[pid]]

    def describe(self) -> List[Dict[str, Any]]:
        outs = []
        for p in self.policies:
            d = {
                "id": p.id,
                "family": p.cfg.family,
                "stochastic": p.cfg.stochastic,
                "noise_sigma": float(p.cfg.noise_sigma),
            }
            if hasattr(p.cfg, "variant"):
                d["variant"] = getattr(p.cfg, "variant")
            if hasattr(p, "regionizer"):
                try:
                    d["regionizer"] = p.regionizer.describe()
                except Exception:
                    pass
            outs.append(d)
        return outs
