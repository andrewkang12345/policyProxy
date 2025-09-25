from __future__ import annotations
import os
import json
import numpy as np
from torch.utils.data import Dataset
from typing import Dict, Any


def quantize_actions(vel: np.ndarray, num_headings: int = 16, num_speeds: int = 3, speed_bins: tuple = (0.0, 0.6, 1.2, 10.0)):
    # vel: [N, agents, 2] -> labels [N] over heading x speed grid (per-agent joint simplified as average heading)
    # For simplicity we quantize the carrier's velocity only (agent 0)
    v = vel[:, 0, :]
    mags = np.linalg.norm(v, axis=-1)
    ang = np.mod(np.arctan2(v[:, 1], v[:, 0]) + 2 * np.pi, 2 * np.pi)
    h = (ang / (2 * np.pi) * num_headings).astype(int)
    s = np.digitize(mags, bins=np.array(speed_bins)) - 1
    s = np.clip(s, 0, num_speeds - 1)
    y = h * num_speeds + s
    return y.astype(int), num_headings * num_speeds


def load_index(index_path: str):
    with open(index_path, "r") as f:
        return json.load(f)


def load_episode(path: str):
    data = np.load(path, allow_pickle=True)
    pos = data["pos"]
    vel = data["vel"]
    lags = data["lags"]
    policy_ids = data.get("policy_ids")
    intents = data.get("intents")
    if policy_ids is None and intents is not None:
        policy_ids = intents
    meta = json.loads(str(data["meta_json"]))
    return {"pos": pos, "vel": vel, "lags": lags, "policy_ids": policy_ids, "intents": intents, "meta": meta}


class NextFrameDataset(Dataset):
    def __init__(self, index_path: str, label_kind: str = "intent"):
        self.index = load_index(index_path)
        # Preload minimal metadata from first file to get W
        first = load_episode(self.index[0]["path"]) if len(self.index) else None
        self.window = int(first["meta"].get("window", 6)) if first else 6
        self.selected_team = int(first["meta"].get("selected_team", 0)) if first else 0
        self.meta: Dict[str, Any] = first["meta"] if first else {}
        self.label_kind = label_kind
        # Build pointers (ep_idx, t) to avoid loading entire dataset at once
        self.ptrs = []
        for ep_i, item in enumerate(self.index):
            ep = load_episode(item["path"])  # small episodes; ok to load
            T = ep["pos"].shape[0]
            for t in range(self.window, T - 1):
                self.ptrs.append((ep_i, t))
        # Cache episodes in memory for simplicity
        self.episodes = [load_episode(it["path"]) for it in self.index]

    def __len__(self):
        return len(self.ptrs)

    def __getitem__(self, idx: int):
        ep_i, t = self.ptrs[idx]
        ep = self.episodes[ep_i]
        W = self.window
        s = ep["pos"][t - W : t]  # [W, teams, agents, 2]
        a = ep["vel"][t, self.selected_team]  # [agents, 2]
        lag = int(ep["lags"][t])
        policy_ids = ep.get("policy_ids")
        if policy_ids is not None:
            intent = int(policy_ids[t])
        else:
            labels = ep.get("intents")
            intent = int(labels[t]) if labels is not None else 0
        if self.label_kind == "policy":
            label = intent
        else:
            labels = ep.get("intents")
            label = int(labels[t]) if labels is not None else intent
        return {
            "state": s.astype(np.float32),
            "action": a.astype(np.float32),
            "lag": lag,
            "intent": intent,
            "policy_id": intent,
            "label": label,
        }

