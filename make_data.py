from __future__ import annotations
import argparse
import os
import json
from copy import deepcopy
import pathlib
import numpy as np
import yaml
from tqdm import trange
import hashlib
from typing import Any

from generator.dataset_generator import DatasetGenerator, GeneratorConfig
from generator.arenas import ArenaConfig
from generator.lag import LagProcessConfig
from tools.distribution import collect_samples, SampleOptions, wasserstein_distance


def _deep_update(base: dict, override: dict) -> dict:
    for key, val in override.items():
        if isinstance(val, dict) and isinstance(base.get(key), dict):
            base[key] = _deep_update(base[key], val)
        else:
            base[key] = deepcopy(val)
    return base


def _load_yaml_with_inherit(path: str) -> dict:
    with open(path, "r") as f:
        data = yaml.safe_load(f) or {}

    inherits = data.pop("inherit", [])
    if isinstance(inherits, str):
        inherits = [inherits]

    merged: dict[str, Any] = {}
    base_dir = pathlib.Path(path).parent
    for parent in inherits:
        parent_path = pathlib.Path(parent)
        if not parent_path.is_absolute():
            parent_path = (base_dir / parent_path).resolve()
        parent_data = _load_yaml_with_inherit(str(parent_path))
        merged = _deep_update(merged, parent_data)

    return _deep_update(merged, data)


def cfg_from_yaml(path: str) -> GeneratorConfig:
    y = _load_yaml_with_inherit(path)

    if isinstance(y.get("generator"), dict):
        merged = deepcopy(y["generator"])
        for key, val in y.items():
            if key == "generator":
                continue
            if isinstance(val, dict) and isinstance(merged.get(key), dict):
                merged[key] = _deep_update(merged[key], val)
            else:
                merged[key] = deepcopy(val)
        y = merged
    # Map nested structures to dataclasses
    arena = ArenaConfig(**y.get("arena", {}))
    lag = LagProcessConfig(**y.get("lag", {}))
    cfg = GeneratorConfig(
        seed=y.get("seed", 0),
        dt=y.get("dt", 0.25),
        steps=y.get("steps", 80),
        window=y.get("window", 6),
        teams=y.get("teams", 2),
        agents_per_team=y.get("agents_per_team", 3),
        selected_team=y.get("selected_team", 0),
        arena=arena,
        policies=y.get("policies"),
        policy_categories=y.get("policy_categories"),
        mixture=y.get("mixture"),
        lag=lag,
        opponent_policies=y.get("opponent_policies"),
        opponent_mixture=y.get("opponent_mixture"),
    )
    # Attach splits and OIDs as aux
    cfg._splits = y.get("splits", {"train_episodes": 64, "val_episodes": 16, "test_episodes": 16})
    cfg._oids = y.get("oids", {})
    cfg._oid_templates = y.get("oid_templates", [])
    cfg._oid_prefix = y.get("oid_prefix")
    if hasattr(cfg, "oid_prefix"):
        setattr(cfg, "oid_prefix", y.get("oid_prefix"))

    extra = {k: v for k, v in y.items() if k not in {
        "seed", "dt", "steps", "window", "teams", "agents_per_team", "selected_team",
        "arena", "policies", "policy_categories", "mixture", "lag", "opponent_policies",
        "opponent_mixture", "splits", "oids", "oid_templates", "oid_prefix"
    }}
    if extra:
        cfg = _apply_generator_overrides(cfg, extra)
    return cfg


def save_episode_npz(path: str, episode: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    meta = dict(episode.get("meta", {}))
    np.savez_compressed(
        path,
        pos=episode["pos"],
        vel=episode["vel"],
        lags=episode["lags"],
        policy_ids=episode["policy_ids"],
        opp_policy_ids=episode.get("opp_policy_ids") if episode.get("opp_policy_ids") is not None else np.array([], dtype=int),
        meta_json=json.dumps(meta),
    )


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def generate_split(out_dir: str, cfg: GeneratorConfig, split_name: str, episodes: int, seed_offset: int = 0):
    index = []
    for e in trange(episodes, desc=f"{split_name}"):
        cfg_ep = deepcopy(cfg)
        cfg_ep.seed = int(cfg.seed + seed_offset + e)
        gen = DatasetGenerator(cfg_ep)
        ep = gen.generate_episode()
        path = os.path.join(out_dir, split_name, f"ep_{e:05d}.npz")
        save_episode_npz(path, ep)
        checksum = sha256_file(path)
        index.append({"path": path, "len": int(ep["pos"].shape[0]), "sha256": checksum})

    idx_path = os.path.join(out_dir, split_name, "index.json")
    os.makedirs(os.path.dirname(idx_path), exist_ok=True)
    with open(idx_path, "w") as f:
        json.dump(index, f, indent=2)
    return idx_path


def _clone_cfg(cfg: GeneratorConfig) -> GeneratorConfig:
    c = deepcopy(cfg)
    return c


def _ensure_bias_on_opponents(cfg: GeneratorConfig, bias: str, severity: float) -> None:
    if cfg.opponent_policies is None:
        cfg.opponent_policies = []
    # enforce random family with bias
    new_list = []
    for i in range(max(2, len(cfg.opponent_policies) or 0)):
        if i < len(cfg.opponent_policies):
            base = dict(cfg.opponent_policies[i])
        else:
            base = {"id": f"op{i}", "family": "random", "stochastic": True, "noise_sigma": 0.1, "regionizer": {"type": "window_hash", "length_scale": 0.25, "centers_per_dim": [2048, 1]}}
        base["family"] = "random"
        base["bias"] = bias
        base["severity"] = float(severity)
        if "regionizer" not in base:
            base["regionizer"] = {"type": "window_hash", "length_scale": 0.25, "centers_per_dim": [2048, 1]}
        new_list.append(base)
    cfg.opponent_policies = new_list


def _with_policy_style(policy: dict[str, Any], style: str, enforce_stochastic: bool | None = None, components: int = 4) -> dict[str, Any]:
    pol = deepcopy(policy)
    if enforce_stochastic is not None:
        pol["stochastic"] = bool(enforce_stochastic)
        if not pol["stochastic"]:
            pol.pop("noise", None)
        else:
            pol.setdefault("noise", {"type": "gaussian", "sigma": 0.08})
    prot = dict(pol.get("prototypes", {}))
    prot["style"] = style
    if pol.get("stochastic"):
        prot.setdefault("components", components)
    else:
        prot.pop("components", None)
    if style == "structured":
        region = dict(pol.get("regionizer", {}))
        bins = region.get("bins_per_dim", [16, 12])
        if isinstance(bins, int):
            bins = [bins, bins]
        region["type"] = "discretized"
        region["bins_per_dim"] = [int(bins[0]), int(bins[1])]
        pol["regionizer"] = region
    pol["prototypes"] = prot
    return pol


def _apply_policy_style(cfg: GeneratorConfig, style: str, enforce_stochastic: bool | None = None, components: int = 4) -> None:
    if cfg.policies is None:
        return
    cfg.policies = [_with_policy_style(p, style, enforce_stochastic, components) for p in cfg.policies]


def tune_ood_to_target(out_dir: str, train_index_path: str, base_cfg: GeneratorConfig, oid_name: str, kind: str, target: float, metric: str = "wasserstein", tol: float = 0.02, max_iters: int = 200, constraint_level: float = 0.08, use_flexible_tuning: bool = True) -> GeneratorConfig:
    """
    Tune OOD generation to hit a target divergence for the specified kind.
    
    For policy shifts: adjusts mixture weights between policies, constrains state+action divergence.
    For state_action shifts: adjusts opponent action bias severity, constrains policy divergence.
    
    constraint_level: maximum allowed divergence for constrained distributions.
    """
    kind = kind.lower()
    if kind not in {"policy", "state_action", "state_only", "action_only"}:
        raise ValueError(f"Unsupported shift_kind '{kind}'")

    tuned = _clone_cfg(base_cfg)
    # Freeze non-target knobs to match base
    if kind == "policy":
        # keep opponents identical
        tuned.opponent_policies = deepcopy(base_cfg.opponent_policies)
        tuned.opponent_mixture = deepcopy(base_cfg.opponent_mixture)
        _apply_policy_style(tuned, "random", enforce_stochastic=None)
    elif kind == "state_action":
        _apply_policy_style(tuned, "structured", enforce_stochastic=True)
    elif kind == "action_only":
        tuned.mixture = deepcopy(base_cfg.mixture)
        _apply_policy_style(tuned, "random", enforce_stochastic=True)
        _ensure_bias_on_opponents(tuned, bias="action_shift", severity=0.0)
    else:
        # keep policy mixture identical
        tuned.mixture = deepcopy(base_cfg.mixture)
        _apply_policy_style(tuned, "random", enforce_stochastic=True)

    # Baseline hists from train split
    arena_size = (tuned.arena.width, tuned.arena.height)
    base_opts = SampleOptions(
        window=tuned.window,
        arena_size=arena_size,
        selected_team=tuned.selected_team,
        limit_episodes=32,
    )
    if kind == "state_action":
        base_samples = {
            "state": collect_samples(train_index_path, "state", base_opts),
            "action": collect_samples(train_index_path, "action", base_opts),
        }
    elif kind == "state_only":
        base_samples = collect_samples(train_index_path, "state", base_opts)
    elif kind == "action_only":
        base_samples = collect_samples(train_index_path, "action", base_opts)
    else:
        base_samples = collect_samples(train_index_path, kind, base_opts)
    
    # Setup constraint checking
    constraint_kinds = []
    if kind == "policy":
        constraint_kinds = ["state", "action"]
    else:
        constraint_kinds = []
    
    constraint_samples = {}
    for ck in constraint_kinds:
        constraint_samples[ck] = collect_samples(train_index_path, ck, base_opts)

    # Initialize control variable
    if kind == "policy":
        # Control: adjust mixture init_weights over two policies
        if tuned.mixture is None:
            tuned.mixture = {"scheduler": "stagnant", "init_weights": [1.0]}
        if not tuned.mixture.get("init_weights") or len(tuned.mixture.get("init_weights")) < 2:
            tuned.mixture["init_weights"] = [0.5, 0.5]
        lo, hi = 0.0, 1.0
        x = 0.5
    else:
        # Control: single knob — opponent action bias severity (same for state/action)
        lo, hi = 0.0, 1.0
        x = 0.0
        bias_kind = "state_shift" if kind == "state_only" else "action_shift"
        _ensure_bias_on_opponents(tuned, bias=bias_kind, severity=x)

    # Choose tuning method (use flexible for both state_action and policy)
    if use_flexible_tuning and kind == "state_action":
        return _tune_flexible_opponent(out_dir, train_index_path, base_cfg, oid_name, kind, target, metric, tol, max_iters, constraint_level, base_samples, constraint_samples, constraint_kinds, arena_size)
    
    # Binary search style tuning with constraint checking  
    pilot_name = f"pilot_{oid_name}"
    for iteration in range(max_iters):
        if kind == "policy":
            tuned.mixture["init_weights"] = [float(1.0 - x), float(x)]
        else:
            bias_kind = "state_shift" if kind == "state_only" else "action_shift"
            _ensure_bias_on_opponents(tuned, bias=bias_kind, severity=x)

        # Small pilot
        pilot_idx = generate_split(out_dir, tuned, pilot_name, episodes=max(4, base_cfg._splits.get("test_episodes", 16) // 8), seed_offset=90_000)
        
        # Compute target divergence(s)
        pilot_opts = SampleOptions(
            window=tuned.window,
            arena_size=arena_size,
            selected_team=tuned.selected_team,
            limit_episodes=32,
        )
        if kind == "state_action":
            state_samples = collect_samples(pilot_idx, "state", pilot_opts)
            action_samples = collect_samples(pilot_idx, "action", pilot_opts)
            d_state = wasserstein_distance(state_samples, base_samples["state"])
            d_action = wasserstein_distance(action_samples, base_samples["action"])
            d = 0.5 * (d_state + d_action)
            target_msg = f"state={d_state:.4f}, action={d_action:.4f}, mean={d:.4f}"
        elif kind == "state_only":
            state_samples = collect_samples(pilot_idx, "state", pilot_opts)
            d = wasserstein_distance(state_samples, base_samples)
            target_msg = f"state={d:.4f}"
        elif kind == "action_only":
            action_samples = collect_samples(pilot_idx, "action", pilot_opts)
            d = wasserstein_distance(action_samples, base_samples)
            target_msg = f"action={d:.4f}"
        else:
            samples = collect_samples(pilot_idx, kind, pilot_opts)
            d = wasserstein_distance(samples, base_samples)
            target_msg = f"{d:.4f}"
        
        # Check constraints
        constraint_satisfied = True
        constraint_divs = {}
        for ck in constraint_kinds:
            pilot_samples = collect_samples(pilot_idx, ck, pilot_opts)
            constraint_div = wasserstein_distance(pilot_samples, constraint_samples[ck])
            constraint_divs[ck] = constraint_div
            if constraint_div > constraint_level:
                constraint_satisfied = False
                break
        
        # Accept if target hit and constraints satisfied
        if abs(d - target) <= tol and constraint_satisfied:
            print(f"  {kind} tuning converged: d={target_msg} (target={target:.4f}), constraints={constraint_divs}")
            break
        
        # If constraints violated, reduce x (less severe shift)
        if not constraint_satisfied:
            hi = x
        else:
            # Constraints OK, adjust based on target
            if d < target:
                lo = x
            else:
                hi = x
        x = 0.5 * (lo + hi)
        
        if iteration == max_iters - 1:
            print(f"  {kind} tuning reached max_iters: d={target_msg} (target={target:.4f}), constraints={constraint_divs}")

    # Clean up pilot index file (keep data to save time)
    # Note: data files are kept to avoid re-generation when close
    return tuned


def _tune_flexible_opponent(out_dir: str, train_index_path: str, base_cfg: GeneratorConfig,
                           oid_name: str, kind: str, target: float, metric: str, tol: float,
                           max_iters: int, constraint_level: float, base_samples: dict,
                           constraint_samples: dict, constraint_kinds: list, arena_size: tuple) -> GeneratorConfig:
    """Flexible opponent tuning using a neural controller for opponent movement."""
    from copy import deepcopy

    rng = np.random.default_rng(base_cfg.seed + 4242)
    hidden_dim = 16
    weight_dim = 1 * hidden_dim + hidden_dim + hidden_dim * 2 + 2
    theta_dim = weight_dim + 5  # scale, amplitude, noise multiplier, mix weight, bias severity
    theta = rng.normal(scale=0.05, size=theta_dim)

    if kind == "policy":
        scale_min, scale_max = 0.3, 3.5
        amp_min, amp_max = 1.5, 10.0
        noise_min, noise_max = 0.1, 1.2
    else:
        scale_min, scale_max = 0.1, 2.0
        amp_min, amp_max = 0.1, 3.0
        noise_min, noise_max = 0.05, 0.8
    phase_template = [float(x) for x in np.linspace(0.0, 2 * np.pi, num=base_cfg.agents_per_team, endpoint=False)]

    penalty_weight = 25.0
    population = 5
    sigma = 0.18
    learning_rate = 0.06
    pilot_name = f"pilot_{oid_name}_nn"
    pilot_seed = 95_000
    pilot_episodes = max(4, base_cfg._splits.get("test_episodes", 16) // 32)

    def _sigmoid(x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-x))

    def unpack(theta_vec: np.ndarray) -> tuple[np.ndarray, float, float, float, float, float]:
        weights = theta_vec[:weight_dim]
        scale_raw = theta_vec[weight_dim]
        amp_raw = theta_vec[weight_dim + 1]
        noise_raw = theta_vec[weight_dim + 2]
        mix_raw = theta_vec[weight_dim + 3]
        severity_raw = theta_vec[weight_dim + 4]
        scale = float(scale_min + (scale_max - scale_min) * _sigmoid(scale_raw))
        amplitude = float(amp_min + (amp_max - amp_min) * _sigmoid(amp_raw))
        noise_mult = float(noise_min + (noise_max - noise_min) * _sigmoid(noise_raw))
        mix = float(_sigmoid(mix_raw))
        severity = float(_sigmoid(severity_raw))
        return weights, scale, amplitude, noise_mult, mix, severity

    def set_amplitude(theta_vec: np.ndarray, amplitude: float) -> np.ndarray:
        theta_new = theta_vec.copy()
        ratio = (amplitude - amp_min) / max(1e-6, (amp_max - amp_min))
        ratio = float(np.clip(ratio, 1e-4, 1 - 1e-4))
        theta_new[weight_dim + 1] = np.log(ratio / (1 - ratio))
        return theta_new

    def build_cfg(theta_vec: np.ndarray) -> GeneratorConfig:
        weights, scale, amplitude, noise_mult, mix, severity = unpack(theta_vec)
        cfg = _clone_cfg(base_cfg)
        cfg.mixture = deepcopy(base_cfg.mixture)
        cfg.opponent_mixture = deepcopy(base_cfg.opponent_mixture)
        if kind == "state_action":
            _apply_policy_style(cfg, "structured", enforce_stochastic=True)
        cfg.opponent_policies = []
        for base_op in base_cfg.opponent_policies or []:
            op_cfg = deepcopy(base_op)
            op_cfg["family"] = "random"
            op_cfg["variant"] = "movement_nn"
            op_cfg["nn_hidden"] = hidden_dim
            op_cfg["nn_params"] = weights.astype(float).tolist()
            op_cfg["nn_scale"] = scale
            op_cfg["nn_amplitude"] = amplitude
            op_cfg["nn_phase"] = phase_template
            op_cfg["nn_mix"] = mix
            op_cfg["stochastic"] = True
            op_cfg["noise_sigma"] = float(base_op.get("noise_sigma", 0.1)) * noise_mult
            if "bias" in op_cfg:
                op_cfg["severity"] = severity
            cfg.opponent_policies.append(op_cfg)
        if not cfg.opponent_policies:
            # Ensure at least one opponent exists for tuning
            cfg.opponent_policies = [{
                "id": "op0",
                "family": "random",
                "variant": "movement_nn",
                "stochastic": True,
                "noise_sigma": 0.1 * noise_mult,
                "nn_hidden": hidden_dim,
                "nn_params": weights.astype(float).tolist(),
                "nn_scale": scale,
                "nn_amplitude": amplitude,
                "nn_phase": phase_template,
                "nn_mix": mix,
                "bias": base_cfg.opponent_policies[0].get("bias") if base_cfg.opponent_policies else None,
                "severity": severity,
            }]
        return cfg

    eval_cache: dict[tuple, tuple[float, dict]] = {}
    random_logs: list[dict[str, Any]] = []
    iteration_logs: list[dict[str, Any]] = []
    calibration_logs: list[dict[str, Any]] = []

    def evaluate(theta_vec: np.ndarray) -> tuple[float, dict]:
        key = tuple(np.round(theta_vec, 6))
        if key in eval_cache:
            cached_obj, cached_info = eval_cache[key]
            return cached_obj, cached_info.copy()

        cfg = build_cfg(theta_vec)
        pilot_idx = generate_split(out_dir, cfg, pilot_name,
                                   episodes=pilot_episodes,
                                   seed_offset=pilot_seed)

        info: dict[str, Any] = {}
        pilot_opts = SampleOptions(
            window=cfg.window,
            arena_size=arena_size,
            selected_team=cfg.selected_team,
            limit_episodes=32,
        )
        if kind == "state_action":
            state_samples = collect_samples(pilot_idx, "state", pilot_opts)
            action_samples = collect_samples(pilot_idx, "action", pilot_opts)
            d_state = float(wasserstein_distance(state_samples, base_samples["state"]))
            d_action = float(wasserstein_distance(action_samples, base_samples["action"]))
            target_metric = max(d_state, d_action)
            info["target_components"] = {"state": d_state, "action": d_action}
        elif kind == "state_only":
            state_samples = collect_samples(pilot_idx, "state", pilot_opts)
            target_metric = float(wasserstein_distance(state_samples, base_samples))
            info["target_components"] = {"state": target_metric}
        else:
            samples = collect_samples(pilot_idx, kind, pilot_opts)
            target_metric = float(wasserstein_distance(samples, base_samples))
            info["target_components"] = {kind: target_metric}

        constraint_penalty = 0.0
        constraint_divs: dict[str, float] = {}
        for ck in constraint_kinds:
            pilot_constraint = collect_samples(pilot_idx, ck, pilot_opts)
            constraint_div = float(wasserstein_distance(pilot_constraint, constraint_samples[ck]))
            constraint_divs[ck] = constraint_div
            if constraint_div > constraint_level:
                constraint_penalty += (constraint_div - constraint_level) ** 2

        obj = abs(target_metric - target) + penalty_weight * constraint_penalty
        weights, scale, amplitude, noise_mult, mix, severity = unpack(theta_vec)
        info.update({
            "divergence": float(target_metric),
            "constraint_divs": {k: float(v) for k, v in constraint_divs.items()},
            "penalty": float(constraint_penalty),
            "scale": float(scale),
            "amplitude": float(amplitude),
            "noise_multiplier": float(noise_mult),
            "weights_norm": float(np.linalg.norm(weights)),
            "mix": float(mix),
            "severity": float(severity),
        })
        eval_cache[key] = (obj, info.copy())
        return obj, info

    print(f"  Starting neural opponent tuning for {kind} with target {target:.4f}")
    best_theta = None
    best_obj = float("inf")
    best_info: dict[str, Any] = {}
    best_penalty = float("inf")

    # Broad random search to find a feasible starting point
    random_samples = max(40, 8 * population)
    for i in range(random_samples):
        candidate = rng.normal(scale=0.4, size=theta_dim)
        obj, info = evaluate(candidate)
        pen = info["penalty"]
        random_logs.append({
            "sample": i + 1,
            "objective": float(obj),
            "divergence": float(info.get("divergence", float("nan"))),
            "penalty": float(pen),
            "constraints": info.get("constraint_divs", {}).copy(),
            "scale": float(info.get("scale", float("nan"))),
            "amplitude": float(info.get("amplitude", float("nan"))),
            "noise_multiplier": float(info.get("noise_multiplier", float("nan"))),
            "mix": float(info.get("mix", float("nan"))),
            "severity": float(info.get("severity", float("nan"))),
        })
        if pen <= 1e-4:
            if obj < best_obj:
                best_obj = obj
                best_theta = candidate.copy()
                best_info = info.copy()
                best_penalty = pen
        elif best_theta is None or pen < best_penalty or (abs(pen - best_penalty) < 1e-6 and obj < best_obj):
            best_obj = obj
            best_theta = candidate.copy()
            best_info = info.copy()
            best_penalty = pen
        if (i + 1) % 10 == 0:
            print(
                f"  Random search {i + 1:02d}/{random_samples}: best_obj={best_obj:.4f}, "
                f"penalty={best_info.get('penalty', 0.0):.4f}, divergence={best_info.get('divergence', 0.0):.4f}"
            )

    if best_theta is None:
        best_theta = theta.copy()
        best_obj, info = evaluate(best_theta)
        best_info = info.copy()

    theta = best_theta.copy()
    print(
        f"  Initial objective: {best_obj:.4f}, divergence={best_info['divergence']:.4f}, "
        f"constraints={best_info['constraint_divs']}, penalty={best_info['penalty']:.4f}"
    )

    no_improve = 0
    for iteration in range(max_iters):
        grad = np.zeros_like(theta)

        for _ in range(population):
            direction = rng.normal(size=theta_dim)
            theta_plus = theta + sigma * direction
            theta_minus = theta - sigma * direction
            obj_plus, info_plus = evaluate(theta_plus)
            obj_minus, info_minus = evaluate(theta_minus)
            grad += (obj_plus - obj_minus) * direction

            if info_plus["penalty"] <= best_info.get("penalty", float("inf")) + 1e-6 and obj_plus < best_obj:
                best_obj, best_info = obj_plus, info_plus.copy()
                best_theta = theta_plus.copy()
                no_improve = 0
            elif obj_plus < best_obj and info_plus["penalty"] < best_info.get("penalty", float("inf")):
                best_obj, best_info = obj_plus, info_plus.copy()
                best_theta = theta_plus.copy()
                no_improve = 0

            if info_minus["penalty"] <= best_info.get("penalty", float("inf")) + 1e-6 and obj_minus < best_obj:
                best_obj, best_info = obj_minus, info_minus.copy()
                best_theta = theta_minus.copy()
                no_improve = 0
            elif obj_minus < best_obj and info_minus["penalty"] < best_info.get("penalty", float("inf")):
                best_obj, best_info = obj_minus, info_minus.copy()
                best_theta = theta_minus.copy()
                no_improve = 0

        grad /= (2.0 * sigma * population)
        theta -= learning_rate * grad

        current_obj, current_info = evaluate(theta)
        if current_obj < best_obj:
            best_obj, best_info = current_obj, current_info.copy()
            best_theta = theta.copy()
            no_improve = 0
        else:
            no_improve += 1

        iteration_logs.append({
            "iteration": iteration + 1,
            "objective": float(current_obj),
            "divergence": float(current_info.get("divergence", float("nan"))),
            "penalty": float(current_info.get("penalty", float("nan"))),
            "constraints": current_info.get("constraint_divs", {}).copy(),
            "scale": float(current_info.get("scale", float("nan"))),
            "amplitude": float(current_info.get("amplitude", float("nan"))),
            "noise_multiplier": float(current_info.get("noise_multiplier", float("nan"))),
            "mix": float(current_info.get("mix", float("nan"))),
            "severity": float(current_info.get("severity", float("nan"))),
        })

        print(
            f"  Iter {iteration + 1:02d}: obj={current_obj:.4f}, divergence={current_info['divergence']:.4f}, "
            f"penalty={current_info['penalty']:.4f}, constraints={current_info['constraint_divs']}"
        )

        if abs(current_info["divergence"] - target) <= tol and current_info["penalty"] <= 1e-4:
            print("  Neural tuning converged within tolerance and constraints.")
            break
        if no_improve >= 6:
            print("  Early stopping due to stagnation.")
            break

        sigma = max(0.05, sigma * 0.97)
        learning_rate = max(0.02, learning_rate * 0.99)

    # Final one-dimensional calibration on amplitude to better match the target
    cal_theta = best_theta.copy()
    cal_info = best_info.copy()
    cal_obj = best_obj
    lo_amp, hi_amp = amp_min, amp_max
    for _ in range(12):
        mid_amp = 0.5 * (lo_amp + hi_amp)
        cand_theta = set_amplitude(best_theta, mid_amp)
        cand_obj, cand_info = evaluate(cand_theta)
        if cand_info["penalty"] <= cal_info.get("penalty", float("inf")) + 1e-6 and cand_obj < cal_obj:
            cal_theta = cand_theta.copy()
            cal_info = cand_info.copy()
            cal_obj = cand_obj
        if cand_info["divergence"] < target:
            lo_amp = mid_amp
        else:
            hi_amp = mid_amp

        calibration_logs.append({
            "amplitude": float(mid_amp),
            "objective": float(cand_obj),
            "divergence": float(cand_info.get("divergence", float("nan"))),
            "penalty": float(cand_info.get("penalty", float("nan"))),
            "constraints": cand_info.get("constraint_divs", {}).copy(),
            "mix": float(cand_info.get("mix", float("nan"))),
            "severity": float(cand_info.get("severity", float("nan"))),
        })

    best_theta = cal_theta
    best_info = cal_info
    best_obj = cal_obj

    final_cfg = build_cfg(best_theta)
    final_pilot = f"final_pilot_{oid_name}_nn"
    final_idx = generate_split(out_dir, final_cfg, final_pilot,
                               episodes=pilot_episodes,
                               seed_offset=pilot_seed + 1)

    final_opts = SampleOptions(
        window=final_cfg.window,
        arena_size=arena_size,
        selected_team=final_cfg.selected_team,
        limit_episodes=32,
    )

    if kind == "state_action":
        state_samples = collect_samples(final_idx, "state", final_opts)
        action_samples = collect_samples(final_idx, "action", final_opts)
        d_state = wasserstein_distance(state_samples, base_samples["state"])
        d_action = wasserstein_distance(action_samples, base_samples["action"])
        d = max(d_state, d_action)
        target_msg = f"state={float(d_state):.4f}, action={float(d_action):.4f}, max={float(d):.4f}"
    elif kind == "state_only":
        state_samples = collect_samples(final_idx, "state", final_opts)
        d = wasserstein_distance(state_samples, base_samples)
        target_msg = f"state={float(d):.4f}"
    else:
        samples = collect_samples(final_idx, kind, final_opts)
        d = wasserstein_distance(samples, base_samples)
        target_msg = f"{float(d):.4f}"

    final_constraints = {}
    for ck in constraint_kinds:
        samples = collect_samples(final_idx, ck, final_opts)
        constraint_div = wasserstein_distance(samples, constraint_samples[ck])
        final_constraints[ck] = float(constraint_div)

    print(
        f"  Final {kind} divergences: {target_msg} (target={target:.4f}), "
        f"constraints={final_constraints}, scale={best_info['scale']:.3f}, amplitude={best_info['amplitude']:.3f}"
    )

    final_components: dict[str, float]
    if kind == "state_action":
        final_components = {"state": float(d_state), "action": float(d_action)}
    else:
        final_components = {kind: float(d)}

    log_payload = {
        "target_kind": kind,
        "target_divergence": float(target),
        "metric": metric,
        "tolerance": float(tol),
        "constraint_level": float(constraint_level),
        "random_search": random_logs,
        "iterations": iteration_logs,
        "calibration": calibration_logs,
        "final": {
            "divergence": final_components,
            "constraints": {k: float(v) for k, v in final_constraints.items()},
            "best_params": {
                "scale": float(best_info.get("scale", float("nan"))),
                "amplitude": float(best_info.get("amplitude", float("nan"))),
                "noise_multiplier": float(best_info.get("noise_multiplier", float("nan"))),
                "weights_norm": float(best_info.get("weights_norm", float("nan"))),
                "mix": float(best_info.get("mix", float("nan"))),
                "severity": float(best_info.get("severity", float("nan"))),
            },
        },
    }

    log_dir = os.path.dirname(final_idx)
    log_path = os.path.join(log_dir, "divergence_log.json")
    try:
        with open(log_path, "w") as f:
            json.dump(log_payload, f, indent=2)
        print(f"  Saved divergence log to {log_path}")
    except Exception as exc:
        print(f"  Warning: failed to write divergence log at {log_path}: {exc}")

    return final_cfg


def _apply_oid_templates(base_cfg: GeneratorConfig) -> dict[str, dict[str, Any]]:
    configured = dict(base_cfg._oids)
    templates = getattr(base_cfg, "_oid_templates", []) or []
    oid_prefix_attr = getattr(base_cfg, "_oid_prefix", None)
    generator_prefix = getattr(base_cfg, "oid_prefix", None)
    if oid_prefix_attr is not None:
        oid_prefix = str(oid_prefix_attr)
    elif generator_prefix is not None:
        oid_prefix = str(generator_prefix)
    else:
        oid_prefix = ""
    if not templates:
        return configured

    seen_labels: set[str] = set()
    for idx, template in enumerate(templates):
        try:
            shift_kind = template["shift_kind"].lower()
            severity = float(template["target"])
        except KeyError as exc:
            raise ValueError(f"oid_templates[{idx}] missing required field {exc}") from exc
        except (TypeError, ValueError) as exc:
            raise ValueError(f"oid_templates[{idx}] target must be numeric: {template.get('target')}") from exc

        metric = template.get("metric", "wasserstein")
        tol = float(template.get("tolerance", 0.02))
        base_label = template.get("label")
        if not base_label:
            base_label = f"{shift_kind}_{int(round(severity * 1000)):03d}"

        label = base_label
        counter = 1
        while label in configured or label in seen_labels:
            label = f"{base_label}_{counter}"
            counter += 1
        seen_labels.add(label)

        configured[label] = {
            "shift_kind": shift_kind,
            "target_divergence": severity,
            "metric": metric,
            "tolerance": tol,
        }

    if oid_prefix:
        renamed: dict[str, dict[str, Any]] = {}
        for label, payload in configured.items():
            renamed[f"{oid_prefix}_{label}" if oid_prefix else label] = payload
        return renamed

    return configured


def _apply_generator_overrides(cfg: GeneratorConfig, overrides: dict[str, Any]) -> GeneratorConfig:
    for key, val in overrides.items():
        if key == "mixture":
            base = deepcopy(cfg.mixture) if cfg.mixture is not None else {}
            cfg.mixture = _deep_update(base, val)
        elif key == "opponent_mixture":
            base = deepcopy(cfg.opponent_mixture) if cfg.opponent_mixture is not None else {}
            cfg.opponent_mixture = _deep_update(base, val)
        elif key in {"policies", "opponent_policies", "policy_categories"}:
            setattr(cfg, key, deepcopy(val))
        elif key == "lag":
            if cfg.lag is None:
                cfg.lag = LagProcessConfig(**val)
            else:
                for kk, vv in val.items():
                    setattr(cfg.lag, kk, vv)
        elif key == "arena":
            if cfg.arena is None:
                cfg.arena = ArenaConfig(**val)
            else:
                for kk, vv in val.items():
                    setattr(cfg.arena, kk, vv)
        elif hasattr(cfg, key):
            setattr(cfg, key, deepcopy(val))
        else:
            print(f"[make_data] Warning: unknown generator override key '{key}' ignored")
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--out", type=str, default="data/v1.0")
    ap.add_argument("--tiny", action="store_true", help="Generate a very small demo dataset")
    ap.add_argument("--sweep", action="store_true", help="Expand oid_templates into individual shift configs and exit")
    ap.add_argument("--dump_splits", type=str, default=None, help="If set, write generated split YAML (per OOD) to this directory instead of running generator")
    args = ap.parse_args()

    cfg = cfg_from_yaml(args.config)
    splits = cfg._splits
    if args.tiny:
        splits = {"train_episodes": 2, "val_episodes": 1, "test_episodes": 1}

    oid_map = _apply_oid_templates(cfg)

    if args.sweep:
        if args.dump_splits:
            os.makedirs(args.dump_splits, exist_ok=True)
        for oid_name, overrides in oid_map.items():
            out_payload = {
                "inherit": args.config,
                "oids": {oid_name: overrides},
            }
            if args.dump_splits:
                out_path = pathlib.Path(args.dump_splits) / f"{oid_name}.yaml"
                with open(out_path, "w") as f:
                    yaml.safe_dump(out_payload, f)
                print(f"Wrote {out_path}")
            else:
                print(yaml.safe_dump(out_payload))
        return

    print("Writing dataset to", args.out)
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "config_used.yaml"), "w") as f:
        yaml.safe_dump({
            "generator": {
                "seed": cfg.seed,
                "dt": cfg.dt,
                "steps": cfg.steps,
                "window": cfg.window,
                "teams": cfg.teams,
                "agents_per_team": cfg.agents_per_team,
                "selected_team": cfg.selected_team,
                "arena": cfg.arena.__dict__,
                "policies": cfg.policies,
                "mixture": cfg.mixture,
                "lag": cfg.lag.__dict__,
                "opponent_policies": cfg.opponent_policies,
                "opponent_mixture": cfg.opponent_mixture,
            },
            "splits": splits,
            "oids": oid_map,
        }, f)

    # IID splits
    generate_split(args.out, cfg, "train", splits.get("train_episodes", 64), seed_offset=0)
    generate_split(args.out, cfg, "val", splits.get("val_episodes", 16), seed_offset=10_000)
    generate_split(args.out, cfg, "test", splits.get("test_episodes", 16), seed_offset=20_000)

    # OOD splits
    for oid_name, overrides in oid_map.items():
        cfg_oid = deepcopy(cfg)
        # Check for targeted shift request
        target = overrides.get("target_divergence")
        kind = overrides.get("shift_kind")  # 'policy' | 'action' | 'state'
        metric = overrides.get("metric", "wasserstein")
        tol = float(overrides.get("tolerance", 0.02))
        train_index_path = os.path.join(args.out, "train", "index.json")
        cfg_tuned = tune_ood_to_target(args.out, train_index_path, cfg, oid_name=oid_name, kind=kind, target=float(target), metric=metric, tol=tol)
        generate_split(args.out, cfg_tuned, f"ood_{oid_name}", max(1, splits.get("test_episodes", 16) // 2), seed_offset=30_000)

    print("Done.")


if __name__ == "__main__":
    main()
