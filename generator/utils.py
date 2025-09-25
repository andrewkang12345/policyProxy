from __future__ import annotations
import numpy as np
from dataclasses import dataclass


def unit(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, eps)


def clip_norm(v: np.ndarray, max_norm: float) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    f = np.minimum(1.0, max_norm / np.maximum(n, 1e-8))
    return v * f


