from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any
import numpy as np


@dataclass
class MixtureConfig:
    scheduler: str = "stagnant"  # stagnant | markov_switch | random_switch
    init_weights: List[float] | None = None
    trans_matrix: List[List[float]] | None = None
    min_dwell_steps: int = 1


class Scheduler:
    def __init__(self, cfg: MixtureConfig, num_policies: int, seed: int):
        self.cfg = cfg
        self.K = int(num_policies)
        self.rng = np.random.default_rng(seed)

    def initial(self) -> int:
        raise NotImplementedError

    def step(self, z: int, t: int) -> int:
        raise NotImplementedError

    def describe(self) -> Dict[str, Any]:
        raise NotImplementedError


class StagnantScheduler(Scheduler):
    def initial(self) -> int:
        if self.cfg.init_weights is None:
            return int(self.rng.integers(self.K))
        w = np.array(self.cfg.init_weights, dtype=float)
        w = w / max(1e-12, w.sum())
        return int(self.rng.choice(self.K, p=w))

    def step(self, z: int, t: int) -> int:
        return z

    def describe(self) -> Dict[str, Any]:
        return {"scheduler": "stagnant", "init_weights": self.cfg.init_weights}


class MarkovSwitchScheduler(Scheduler):
    def __init__(self, cfg: MixtureConfig, num_policies: int, seed: int):
        super().__init__(cfg, num_policies, seed)
        if cfg.trans_matrix is None:
            P = np.eye(self.K) * 0.95 + (np.ones((self.K, self.K)) - np.eye(self.K)) * (0.05 / max(1, self.K - 1))
            self.P = P
        else:
            self.P = np.array(cfg.trans_matrix, dtype=float)
        self.dwell_left = int(cfg.min_dwell_steps)

    def initial(self) -> int:
        if self.cfg.init_weights is None:
            return int(self.rng.integers(self.K))
        w = np.array(self.cfg.init_weights, dtype=float)
        w = w / max(1e-12, w.sum())
        z0 = int(self.rng.choice(self.K, p=w))
        self.dwell_left = int(self.cfg.min_dwell_steps)
        return z0

    def step(self, z: int, t: int) -> int:
        if self.dwell_left > 0:
            self.dwell_left -= 1
            return z
        p = self.P[z]
        z_next = int(self.rng.choice(self.K, p=p))
        if z_next != z:
            self.dwell_left = int(self.cfg.min_dwell_steps)
        return z_next

    def describe(self) -> Dict[str, Any]:
        return {
            "scheduler": "markov_switch",
            "trans_matrix": self.P.tolist(),
            "min_dwell_steps": int(self.cfg.min_dwell_steps),
            "init_weights": self.cfg.init_weights,
        }


class RandomSwitchScheduler(Scheduler):
    def __init__(self, cfg: MixtureConfig, num_policies: int, seed: int):
        super().__init__(cfg, num_policies, seed)
        self.min_dwell = int(cfg.min_dwell_steps)
        self.dwell_left = self.min_dwell

    def initial(self) -> int:
        if self.cfg.init_weights is None:
            z0 = int(self.rng.integers(self.K))
        else:
            w = np.array(self.cfg.init_weights, dtype=float)
            w = w / max(1e-12, w.sum())
            z0 = int(self.rng.choice(self.K, p=w))
        self.dwell_left = self.min_dwell
        return z0

    def step(self, z: int, t: int) -> int:
        if self.dwell_left > 0:
            self.dwell_left -= 1
            return z
        # Uniformly pick a different policy
        choices = [i for i in range(self.K) if i != z]
        z_next = int(self.rng.choice(choices)) if choices else z
        if z_next != z:
            self.dwell_left = self.min_dwell
        return z_next

    def describe(self) -> Dict[str, Any]:
        return {
            "scheduler": "random_switch",
            "min_dwell_steps": int(self.min_dwell),
            "init_weights": self.cfg.init_weights,
        }


def build_scheduler(cfg: MixtureConfig, num_policies: int, seed: int) -> Scheduler:
    if cfg.scheduler == "stagnant":
        return StagnantScheduler(cfg, num_policies, seed)
    elif cfg.scheduler == "markov_switch":
        return MarkovSwitchScheduler(cfg, num_policies, seed)
    elif cfg.scheduler == "random_switch":
        return RandomSwitchScheduler(cfg, num_policies, seed)
    else:
        raise ValueError(f"Unknown scheduler: {cfg.scheduler}")


