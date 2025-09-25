from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, List
import json
import os
import numpy as np


@dataclass
class SampleOptions:
    window: int
    arena_size: Tuple[float, float]
    selected_team: int = 0
    limit_episodes: int = 32
    max_samples: int = 8192


def _normalize_positions(win_pos: np.ndarray, arena_size: Tuple[float, float]) -> np.ndarray:
    out = win_pos.astype(float, copy=True)
    w = max(1e-6, float(arena_size[0]))
    h = max(1e-6, float(arena_size[1]))
    out[..., 0] /= w
    out[..., 1] /= h
    return out


def _load_index(index_path: str, limit: int | None) -> List[str]:
    with open(index_path, "r") as f:
        index = json.load(f)
    paths = [entry["path"] if isinstance(entry, dict) else entry["path"] for entry in index]
    if limit is not None:
        paths = paths[:limit]
    return paths


def collect_state_samples(index_path: str, opts: SampleOptions) -> np.ndarray:
    paths = _load_index(index_path, opts.limit_episodes)
    samples: List[np.ndarray] = []
    for p in paths:
        data = np.load(p)
        pos = data["pos"]  # [T, teams, agents, 2]
        T, teams, agents, _ = pos.shape
        for t in range(T):
            start = max(0, t - opts.window + 1)
            win = pos[start : t + 1]
            if win.shape[0] < opts.window:
                pad = np.repeat(win[:1], opts.window - win.shape[0], axis=0)
                win = np.concatenate([pad, win], axis=0)
            norm = _normalize_positions(win, opts.arena_size)
            samples.append(norm.reshape(opts.window * teams * agents * 2))
    if not samples:
        return np.zeros((0, opts.window * 2), dtype=float)
    stack = np.vstack(samples).astype(float)
    if stack.shape[0] > opts.max_samples:
        rng = np.random.default_rng(1234)
        idx = rng.choice(stack.shape[0], size=opts.max_samples, replace=False)
        stack = stack[idx]
    return stack


def collect_action_samples(index_path: str, opts: SampleOptions) -> np.ndarray:
    paths = _load_index(index_path, opts.limit_episodes)
    samples: List[np.ndarray] = []
    for p in paths:
        data = np.load(p)
        vel = data["vel"]  # [T, teams, agents, 2]
        team_actions = vel[:, opts.selected_team]  # [T, agents, 2]
        samples.extend(team_actions.reshape(team_actions.shape[0], -1))
    if not samples:
        return np.zeros((0, 2), dtype=float)
    stack = np.vstack(samples).astype(float)
    if stack.shape[0] > opts.max_samples:
        rng = np.random.default_rng(4321)
        idx = rng.choice(stack.shape[0], size=opts.max_samples, replace=False)
        stack = stack[idx]
    return stack


def collect_policy_samples(index_path: str, opts: SampleOptions) -> np.ndarray:
    paths = _load_index(index_path, opts.limit_episodes)
    values: List[np.ndarray] = []
    for p in paths:
        data = np.load(p)
        ids = data["policy_ids"].astype(float)
        values.append(ids[:, None])
    if not values:
        return np.zeros((0, 1), dtype=float)
    stack = np.vstack(values)
    if stack.shape[0] > opts.max_samples:
        rng = np.random.default_rng(2468)
        idx = rng.choice(stack.shape[0], size=opts.max_samples, replace=False)
        stack = stack[idx]
    return stack


def collect_samples(index_path: str, kind: str, opts: SampleOptions) -> np.ndarray:
    kind = kind.lower()
    if kind == "state":
        return collect_state_samples(index_path, opts)
    if kind == "action":
        return collect_action_samples(index_path, opts)
    if kind == "policy":
        return collect_policy_samples(index_path, opts)
    raise ValueError(f"Unknown sample kind: {kind}")


def wasserstein_1d(x: np.ndarray, y: np.ndarray) -> float:
    if x.size == 0 or y.size == 0:
        return 0.0
    x = np.sort(x.astype(float))
    y = np.sort(y.astype(float))
    n = x.size
    m = y.size
    if n == m:
        return float(np.mean(np.abs(x - y)))
    grid = np.linspace(0.0, 1.0, num=max(n, m), endpoint=True)
    xq = np.quantile(x, grid)
    yq = np.quantile(y, grid)
    return float(np.mean(np.abs(xq - yq)))


def wasserstein_distance(samples_a: np.ndarray, samples_b: np.ndarray) -> float:
    if samples_a.size == 0 or samples_b.size == 0:
        return 0.0
    a = samples_a.reshape(samples_a.shape[0], -1)
    b = samples_b.reshape(samples_b.shape[0], -1)
    dims = a.shape[1]
    total = 0.0
    for d in range(dims):
        total += wasserstein_1d(a[:, d], b[:, d])
    return total / float(dims)

