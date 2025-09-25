from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Dict, Any
import numpy as np


def _ensure_tuple2(x) -> Tuple[int, int]:
    if isinstance(x, (list, tuple)) and len(x) == 2:
        return int(x[0]), int(x[1])
    return int(x), int(x)


@dataclass
class RegionizerConfig:
    type: str = "discretized"  # "discretized" | "smoothed"
    bins_per_dim: Tuple[int, int] | int = (12, 12)  # for discretized
    tie_break: str = "nearest"
    smoother: str = "rbf"  # for smoothed
    length_scale: float = 0.8
    centers_per_dim: Tuple[int, int] | int = (6, 4)


class Regionizer:
    def state_feature(self, pos: list[np.ndarray], selected_team: int, arena_size: Tuple[float, float]) -> np.ndarray:
        # Default feature: normalized centroid (x, y) of the selected team
        team_pos = np.array(pos[selected_team])
        centroid = team_pos.mean(axis=0)  # [2]
        w, h = float(arena_size[0]), float(arena_size[1])
        feat = np.array([centroid[0] / max(1e-6, w), centroid[1] / max(1e-6, h)], dtype=float)
        return feat  # [2] in [0,1]^2

    def state_feature_window(self, win_pos: np.ndarray, selected_team: int, arena_size: Tuple[float, float]) -> np.ndarray:
        # win_pos: [W, teams, agents, 2]; default reduces to last-frame centroid
        last = win_pos[-1]
        return self.state_feature([last[i] for i in range(last.shape[0])], selected_team, arena_size)

    def describe(self) -> Dict[str, Any]:
        raise NotImplementedError


class DiscretizedRegionizer(Regionizer):
    def __init__(self, cfg: RegionizerConfig):
        self.cfg = cfg
        self.bins_x, self.bins_y = _ensure_tuple2(cfg.bins_per_dim)

    def bin_index(self, feat_xy: np.ndarray) -> Tuple[int, int, int]:
        # feat_xy in [0,1]^2
        bx = int(np.clip(np.floor(feat_xy[0] * self.bins_x), 0, self.bins_x - 1))
        by = int(np.clip(np.floor(feat_xy[1] * self.bins_y), 0, self.bins_y - 1))
        flat = by * self.bins_x + bx
        return bx, by, flat

    def bin_center(self, bx: int, by: int) -> np.ndarray:
        cx = (bx + 0.5) / max(1, self.bins_x)
        cy = (by + 0.5) / max(1, self.bins_y)
        return np.array([cx, cy], dtype=float)

    def center_from_index(self, flat: int) -> np.ndarray:
        bx = int(flat % self.bins_x)
        by = int(flat // self.bins_x)
        return self.bin_center(bx, by)

    def describe(self) -> Dict[str, Any]:
        return {
            "type": "discretized",
            "bins_per_dim": [self.bins_x, self.bins_y],
            "tie_break": self.cfg.tie_break,
        }


class SmoothedRegionizer(Regionizer):
    def __init__(self, cfg: RegionizerConfig):
        self.cfg = cfg
        self.cx, self.cy = _ensure_tuple2(cfg.centers_per_dim)
        # Create RBF centers on a grid in [0,1]^2
        xs = np.linspace(0.0, 1.0, self.cx)
        ys = np.linspace(0.0, 1.0, self.cy)
        mx, my = np.meshgrid(xs, ys, indexing="xy")
        self.centers = np.stack([mx.ravel(), my.ravel()], axis=-1)  # [C,2]
        self.length_scale = float(cfg.length_scale)

    def rbf_phi(self, feat_xy: np.ndarray) -> np.ndarray:
        # Radial basis features over normalized centroid
        dif = self.centers - feat_xy[None, :]
        d2 = np.sum(dif * dif, axis=-1)
        phi = np.exp(-0.5 * d2 / max(1e-12, self.length_scale ** 2))  # [C]
        # Normalize to sum 1 for stability
        s = float(np.sum(phi))
        return phi / max(1e-8, s)

    def describe(self) -> Dict[str, Any]:
        return {
            "type": "smoothed",
            "smoother": self.cfg.smoother,
            "length_scale": self.length_scale,
            "centers_per_dim": [self.cx, self.cy],
        }


class WindowHashRegionizer(Regionizer):
    def __init__(self, cfg: RegionizerConfig):
        # Reuse length_scale as quantization granularity proxy; centers_per_dim first value as num_buckets if provided
        self.quant = float(getattr(cfg, "length_scale", 0.25))
        nb = getattr(cfg, "centers_per_dim", (256, 1))
        if isinstance(nb, (list, tuple)):
            self.num_buckets = int(nb[0])
        else:
            self.num_buckets = int(nb)

    def _quantize(self, win_pos: np.ndarray, arena_size: Tuple[float, float]) -> np.ndarray:
        # Normalize to [0,1] and quantize
        w, h = float(arena_size[0]), float(arena_size[1])
        norm = win_pos.copy()
        norm[..., 0] = norm[..., 0] / max(1e-6, w)
        norm[..., 1] = norm[..., 1] / max(1e-6, h)
        q = np.floor(norm / max(1e-6, self.quant))
        return q.astype(int)

    def bin_index_from_window(self, win_pos: np.ndarray, selected_team: int, arena_size: Tuple[float, float]) -> int:
        # Use all teams' positions over window; stable hash into [0, num_buckets)
        q = self._quantize(win_pos, arena_size)
        # Mix bits: xor-reduce over axes for stability to ordering changes
        h = 0
        flat = q.reshape(-1)
        for v in flat:
            h = ((h << 5) - h) ^ int(v)
            h &= 0xFFFFFFFF
        return int(h % max(1, self.num_buckets))

    def describe(self) -> Dict[str, Any]:
        return {
            "type": "window_hash",
            "quant": self.quant,
            "num_buckets": int(self.num_buckets),
        }


def build_regionizer(cfg: RegionizerConfig) -> Regionizer:
    if cfg.type == "discretized":
        return DiscretizedRegionizer(cfg)
    elif cfg.type == "smoothed":
        return SmoothedRegionizer(cfg)
    elif cfg.type == "window_hash":
        return WindowHashRegionizer(cfg)
    else:
        raise ValueError(f"Unknown regionizer type: {cfg.type}")

