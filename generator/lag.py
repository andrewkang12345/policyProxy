from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class LagProcessConfig:
    mode: str = "random"  # "fixed", "random", "state_dependent"
    fixed_k: int = 2
    random_min: int = 0
    random_max: int = 4
    sd_small_k: int = 1
    sd_large_k: int = 4
    sd_threshold: float = 4.0  # if min inter-team distance < threshold -> small lag
    per_agent_hetero: bool = False


class LagProcess:
    def __init__(self, cfg: LagProcessConfig, rng: np.random.Generator):
        self.cfg = cfg
        self.rng = rng

    def sample(self, context_state: dict | None = None, agent_count: int = 1):
        c = self.cfg
        if c.mode == "fixed":
            if c.per_agent_hetero:
                return np.full(agent_count, c.fixed_k, dtype=int)
            return c.fixed_k
        elif c.mode == "random":
            if c.per_agent_hetero:
                return self.rng.integers(c.random_min, c.random_max + 1, size=agent_count)
            return int(self.rng.integers(c.random_min, c.random_max + 1))
        elif c.mode == "state_dependent":
            # use min inter-team distance as a simple proxy
            if context_state is None or "min_interteam_dist" not in context_state:
                val = c.sd_large_k
            else:
                val = c.sd_small_k if context_state["min_interteam_dist"] < c.sd_threshold else c.sd_large_k
            if c.per_agent_hetero:
                return np.full(agent_count, val, dtype=int)
            return int(val)
        else:
            raise ValueError(f"Unknown lag mode: {c.mode}")

