from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class ArenaConfig:
    width: float = 20.0
    height: float = 14.0
    obstacle_count: int = 0
    obstacle_radius: float = 1.0
    obstacle_seed: int = 0


class Arena:
    def __init__(self, cfg: ArenaConfig):
        self.cfg = cfg
        self.width = cfg.width
        self.height = cfg.height
        self.obstacles = self._gen_obstacles()

    def _gen_obstacles(self):
        if self.cfg.obstacle_count <= 0:
            return []
        rng = np.random.default_rng(self.cfg.obstacle_seed)
        obs = []
        for _ in range(self.cfg.obstacle_count):
            x = rng.uniform(1.0, self.width - 1.0)
            y = rng.uniform(1.0, self.height - 1.0)
            r = self.cfg.obstacle_radius
            obs.append((x, y, r))
        return obs

    def reflect_bounds(self, pos: np.ndarray, vel: np.ndarray):
        # pos, vel: [N,2]
        x, y = pos[:, 0], pos[:, 1]
        hit_x_lo = x < 0.0
        hit_x_hi = x > self.width
        hit_y_lo = y < 0.0
        hit_y_hi = y > self.height
        vel[hit_x_lo | hit_x_hi, 0] *= -1.0
        vel[hit_y_lo | hit_y_hi, 1] *= -1.0
        pos[:, 0] = np.clip(pos[:, 0], 0.0, self.width)
        pos[:, 1] = np.clip(pos[:, 1], 0.0, self.height)
        return pos, vel

    def obstacle_repulsion(self, pos: np.ndarray, strength: float = 5.0, radius: float = 1.5):
        if not self.obstacles:
            return np.zeros_like(pos)
        acc = np.zeros_like(pos)
        for (ox, oy, r) in self.obstacles:
            o = np.array([ox, oy])[None, :]
            d = pos - o
            dist = np.linalg.norm(d, axis=-1, keepdims=True)
            mask = (dist < (r + radius))
            # Soft repulsion when too close
            push = strength * (1.0 / np.maximum(dist, 1e-3) - 1.0 / (r + radius))
            push = np.clip(push, 0.0, strength)
            acc += np.where(mask, d / np.maximum(dist, 1e-3) * push, 0.0)
        return acc

