from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple
import numpy as np

from .regionizers import RegionizerConfig, build_regionizer, DiscretizedRegionizer, SmoothedRegionizer, WindowHashRegionizer

_DEFAULT_STOCHASTIC_COMPONENTS = 4


@dataclass
class NoiseConfig:
    type: str = "gaussian"  # gaussian | none
    sigma: float = 0.0
    anneal: float = 0.0  # fraction per step decrease; 0 for constant


@dataclass
class PolicyConfig:
    id: str
    family: str = "proto_actions"  # currently only proto_actions
    stochastic: bool = False
    noise: Dict[str, Any] | None = None
    regionizer: Dict[str, Any] | None = None
    # prototype store sizing; if None, inferred from regionizer config
    prototypes: Dict[str, Any] | None = None
    # lag knob: whether to use generator-provided lagged state or current snapshot; optional extra lag frames
    use_lagged_state: bool = True
    extra_lag_k: int = 0


class Policy:
    def __init__(self, cfg: PolicyConfig, agents: int, arena_size: Tuple[float, float], seed: int):
        self.cfg = cfg
        self.id = cfg.id
        self.agents = int(agents)
        self.arena_size = (float(arena_size[0]), float(arena_size[1]))
        self.rng = np.random.default_rng(seed)
        rcfg = RegionizerConfig(**(cfg.regionizer or {}))
        self.regionizer = build_regionizer(rcfg)
        ncfg = NoiseConfig(**(cfg.noise or {})) if cfg.stochastic else NoiseConfig(type="none", sigma=0.0)
        self.noise_cfg = ncfg
        

    def step(self, world_snap: Dict[str, Any], selected_team: int, t: int) -> np.ndarray:
        raise NotImplementedError

    def describe(self) -> Dict[str, Any]:
        raise NotImplementedError


class ProtoActionsPolicy(Policy):
    def __init__(self, cfg: PolicyConfig, agents: int, arena_size: Tuple[float, float], seed: int):
        super().__init__(cfg, agents, arena_size, seed)
        # Initialize prototype actions
        if isinstance(self.regionizer, DiscretizedRegionizer):
            bins_x, bins_y = self.regionizer.bins_x, self.regionizer.bins_y
            K = bins_x * bins_y
        elif isinstance(self.regionizer, SmoothedRegionizer):
            K = self.regionizer.centers.shape[0]
        else:
            K = 1

        proto_cfg = dict(self.cfg.prototypes or {})
        style = proto_cfg.get("style", "random")
        self.components = int(proto_cfg.get("components", _DEFAULT_STOCHASTIC_COMPONENTS if cfg.stochastic else 1))
        self.components = max(1, self.components)

        # Initialize prototypes [K, components, agents, 2]
        self.prototypes = np.zeros((K, self.components, self.agents, 2), dtype=float)
        self.comp_weights = np.zeros((K, self.components), dtype=float)

        def _random_unit_vec() -> np.ndarray:
            v = self.rng.normal(size=2)
            n = np.linalg.norm(v)
            if n < 1e-6:
                return np.array([1.0, 0.0], dtype=float)
            return v / n

        def _structured_vec(center: np.ndarray, agent_idx: int, comp_idx: int) -> np.ndarray:
            # Base direction points outward from arena center; rotate per agent/component for diversity
            base = np.array([center[0] - 0.5, center[1] - 0.5], dtype=float)
            n = np.linalg.norm(base)
            if n < 1e-6:
                base = np.array([0.0, 1.0], dtype=float)
            else:
                base = base / n
            # Agent-specific rotation plus component jitter
            angle = (agent_idx * 0.6) + (comp_idx - (self.components - 1) / 2.0) * 0.4
            ca, sa = np.cos(angle), np.sin(angle)
            rot = np.array([[ca, -sa], [sa, ca]], dtype=float)
            return (rot @ base).astype(float)

        predefined = None
        if "values" in proto_cfg:
            arr = np.array(proto_cfg["values"], dtype=float)
            if arr.ndim == 4:
                if arr.shape[0] == K and arr.shape[1] == self.components and arr.shape[2] == self.agents and arr.shape[3] == 2:
                    predefined = arr

        for k in range(K):
            center = None
            if isinstance(self.regionizer, DiscretizedRegionizer):
                center = self.regionizer.center_from_index(k)
            elif isinstance(self.regionizer, SmoothedRegionizer):
                center = self.regionizer.centers[k]
            for c in range(self.components):
                for a in range(self.agents):
                    if predefined is not None:
                        vec = predefined[k, c, a]
                    elif style == "structured" and center is not None:
                        vec = _structured_vec(center, a, c)
                    else:
                        vec = _random_unit_vec()
                    n = np.linalg.norm(vec)
                    if n < 1e-6:
                        vec = np.array([1.0, 0.0], dtype=float)
                    else:
                        vec = vec / n
                    self.prototypes[k, c, a] = vec
            if self.cfg.stochastic and self.components > 1:
                self.comp_weights[k] = self.rng.dirichlet(np.ones(self.components, dtype=float))
            else:
                self.comp_weights[k] = np.ones(self.components, dtype=float)

    def _noise_sigma(self, t: int) -> float:
        if self.noise_cfg.type != "gaussian":
            return 0.0
        if self.noise_cfg.anneal and self.noise_cfg.anneal > 0.0:
            return max(0.0, float(self.noise_cfg.sigma) * (1.0 - self.noise_cfg.anneal) ** t)
        return float(self.noise_cfg.sigma)

    def step(self, world_snap: Dict[str, Any], selected_team: int, t: int) -> np.ndarray:
        pos = world_snap["pos"]
        # Prefer window feature if present for all regionizers
        if "win_pos" in world_snap:
            win_pos = world_snap["win_pos"]
            if isinstance(self.regionizer, WindowHashRegionizer):
                idx = self.regionizer.bin_index_from_window(win_pos, selected_team, self.arena_size)
                flat_idx = idx % self.prototypes.shape[0]
                weights = self.comp_weights[flat_idx]
                weights = weights / max(1e-6, weights.sum())
                comp = 0
                if self.cfg.stochastic and self.components > 1:
                    comp = int(self.rng.choice(self.components, p=weights))
                mean = self.prototypes[flat_idx, comp]
            else:
                # Use regionizer's window feature when available (defaults to last-frame centroid)
                feat = self.regionizer.state_feature_window(win_pos, selected_team, self.arena_size)
                if isinstance(self.regionizer, DiscretizedRegionizer):
                    _, _, idx = self.regionizer.bin_index(feat)
                    weights = self.comp_weights[idx]
                    weights = weights / max(1e-6, weights.sum())
                    comp = 0
                    if self.cfg.stochastic and self.components > 1:
                        comp = int(self.rng.choice(self.components, p=weights))
                    mean = self.prototypes[idx, comp]
                elif isinstance(self.regionizer, SmoothedRegionizer):
                    phi = self.regionizer.rbf_phi(feat)
                    weighted = (phi[:, None, None, None] * self.prototypes).sum(axis=0)
                    if self.cfg.stochastic and self.components > 1:
                        weights = (phi[:, None] * self.comp_weights).sum(axis=0)
                        weights = weights / max(1e-6, weights.sum())
                        comp = int(self.rng.choice(self.components, p=weights))
                    else:
                        comp = 0
                    mean = weighted[comp]
                else:
                    weights = self.comp_weights[0]
                    weights = weights / max(1e-6, weights.sum())
                    comp = 0
                    if self.cfg.stochastic and self.components > 1:
                        comp = int(self.rng.choice(self.components, p=weights))
                    mean = self.prototypes[0, comp]
        else:
            feat = self.regionizer.state_feature(pos, selected_team, self.arena_size)  # [2]
            if isinstance(self.regionizer, DiscretizedRegionizer):
                _, _, idx = self.regionizer.bin_index(feat)
                weights = self.comp_weights[idx]
                weights = weights / max(1e-6, weights.sum())
                comp = 0
                if self.cfg.stochastic and self.components > 1:
                    comp = int(self.rng.choice(self.components, p=weights))
                mean = self.prototypes[idx, comp]
            elif isinstance(self.regionizer, SmoothedRegionizer):
                phi = self.regionizer.rbf_phi(feat)  # [K]
                weighted = (phi[:, None, None, None] * self.prototypes).sum(axis=0)
                if self.cfg.stochastic and self.components > 1:
                    weights = (phi[:, None] * self.comp_weights).sum(axis=0)
                    weights = weights / max(1e-6, weights.sum())
                    comp = int(self.rng.choice(self.components, p=weights))
                else:
                    comp = 0
                mean = weighted[comp]
            else:
                weights = self.comp_weights[0]
                weights = weights / max(1e-6, weights.sum())
                comp = 0
                if self.cfg.stochastic and self.components > 1:
                    comp = int(self.rng.choice(self.components, p=weights))
                mean = self.prototypes[0, comp]

        act = mean.copy()
        if self.cfg.stochastic and self.noise_cfg.type == "gaussian":
            sigma = self._noise_sigma(t)
            if sigma > 0:
                act = act + self.rng.normal(scale=sigma, size=act.shape)
        return act

    def describe(self) -> Dict[str, Any]:
        d = {
            "id": self.id,
            "family": self.cfg.family,
            "stochastic": self.cfg.stochastic,
            "noise": self.noise_cfg.__dict__,
            "regionizer": self.regionizer.describe(),
            "prototypes_shape": list(self.prototypes.shape),
            "components": int(self.components),
            "prototype_style": (self.cfg.prototypes or {}).get("style", "random"),
        }
        return d


def build_policy(cfg: PolicyConfig, agents: int, arena_size: Tuple[float, float], seed: int) -> Policy:
    family = cfg.family
    if family == "proto_actions":
        return ProtoActionsPolicy(cfg, agents, arena_size, seed)
    else:
        raise ValueError(f"Unknown policy family: {family}")


class PolicyManager:
    def __init__(self, policy_cfgs: List[Dict[str, Any]], agents: int, arena_size: Tuple[float, float], seed: int):
        self.policies: List[Policy] = []
        self.id_to_index: Dict[str, int] = {}
        for i, pcfg in enumerate(policy_cfgs):
            cfg = PolicyConfig(**pcfg)
            pol = build_policy(cfg, agents, arena_size, seed + i * 97)
            self.policies.append(pol)
            self.id_to_index[cfg.id] = i

    def get_by_id(self, pid: str) -> Policy:
        return self.policies[self.id_to_index[pid]]

    def describe(self) -> List[Dict[str, Any]]:
        return [p.describe() for p in self.policies]
