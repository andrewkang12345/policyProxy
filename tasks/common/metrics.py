from __future__ import annotations
import numpy as np


def ade(pred_traj: np.ndarray, gt_traj: np.ndarray) -> float:
    # pred, gt: [N, agents, 2]
    d = np.linalg.norm(pred_traj - gt_traj, axis=-1)
    return float(d.mean())


def fde(pred_traj: np.ndarray, gt_traj: np.ndarray) -> float:
    # Same as ADE in one-step setting
    d = np.linalg.norm(pred_traj - gt_traj, axis=-1)
    return float(d.mean())


def discrete_topk(logits: np.ndarray, labels: np.ndarray, k: int = 5) -> float:
    # logits: [N, C], labels: [N]
    idx = np.argsort(-logits, axis=-1)[:, :k]
    ok = (idx == labels[:, None]).any(axis=-1)
    return float(ok.mean())


# Removed unused metrics: mae, rmse, gaussian_nll


def brier_score(prob: np.ndarray, labels: np.ndarray) -> float:
    # prob: [N, C] probabilities; labels: [N] ints
    N, C = prob.shape
    one_hot = np.zeros((N, C), dtype=float)
    one_hot[np.arange(N), labels] = 1.0
    return float(np.mean((prob - one_hot) ** 2))


def ece(prob: np.ndarray, labels: np.ndarray, num_bins: int = 15) -> float:
    # Expected Calibration Error for multi-class softmax probabilities
    conf = prob.max(axis=1)
    pred = prob.argmax(axis=1)
    correct = (pred == labels).astype(float)
    bins = np.linspace(0.0, 1.0, num_bins + 1)
    ece_val = 0.0
    for b in range(num_bins):
        lo, hi = bins[b], bins[b + 1]
        mask = (conf > lo) & (conf <= hi) if b > 0 else (conf >= lo) & (conf <= hi)
        if not np.any(mask):
            continue
        acc = float(correct[mask].mean())
        avg_conf = float(conf[mask].mean())
        frac = float(mask.mean())
        ece_val += frac * abs(acc - avg_conf)
    return float(ece_val)


def reliability_bins(prob: np.ndarray, labels: np.ndarray, num_bins: int = 15):
    # Returns (bin_centers, avg_conf, avg_acc) for plotting
    conf = prob.max(axis=1)
    pred = prob.argmax(axis=1)
    correct = (pred == labels).astype(float)
    bins = np.linspace(0.0, 1.0, num_bins + 1)
    centers = (bins[:-1] + bins[1:]) / 2.0
    avg_conf = np.zeros(num_bins, dtype=float)
    avg_acc = np.zeros(num_bins, dtype=float)
    for b in range(num_bins):
        lo, hi = bins[b], bins[b + 1]
        mask = (conf > lo) & (conf <= hi) if b > 0 else (conf >= lo) & (conf <= hi)
        if np.any(mask):
            avg_conf[b] = float(conf[mask].mean())
            avg_acc[b] = float(correct[mask].mean())
        else:
            avg_conf[b] = (lo + hi) / 2.0
            avg_acc[b] = np.nan
    return centers, avg_conf, avg_acc

