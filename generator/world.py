from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .arenas import Arena
from .utils import clip_norm


@dataclass
class TeamConfig:
    agent_count: int = 3
    max_speed: float = 1.2
    act_noise: float = 0.05


class World:
    def __init__(self, arena: Arena, team_cfgs: list[TeamConfig], dt: float = 0.25, seed: int = 0):
        self.arena = arena
        self.team_cfgs = team_cfgs
        self.team_count = len(team_cfgs)
        self.dt = dt
        self.rng = np.random.default_rng(seed)

        self.pos = []  # list of [N_i, 2]
        self.vel = []  # list of [N_i, 2]
        for cfg in team_cfgs:
            n = cfg.agent_count
            x = self.rng.uniform(2.0, arena.width - 2.0, size=(n,))
            y = self.rng.uniform(2.0, arena.height - 2.0, size=(n,))
            self.pos.append(np.stack([x, y], axis=-1))
            self.vel.append(np.zeros((n, 2), dtype=float))

    def step(self, actions: list[np.ndarray]):
        # actions: list of [N_i, 2] velocities per team
        for i, act in enumerate(actions):
            cfg = self.team_cfgs[i]
            a = clip_norm(act + self.rng.normal(scale=cfg.act_noise, size=act.shape), cfg.max_speed)
            self.vel[i] = a

        # Integrate
        for i in range(self.team_count):
            self.pos[i] = self.pos[i] + self.vel[i] * self.dt
            # Reflect at boundaries
            self.pos[i], self.vel[i] = self.arena.reflect_bounds(self.pos[i], self.vel[i])

        # Simple obstacle repulsion (applied as velocity perturbation)
        for i in range(self.team_count):
            rep = self.arena.obstacle_repulsion(self.pos[i])
            self.vel[i] += rep * 0.0  # turned off by default

        # Return snapshot
        return {
            "pos": [p.copy() for p in self.pos],
            "vel": [v.copy() for v in self.vel],
        }

