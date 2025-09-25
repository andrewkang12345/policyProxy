# Policy-or-Proxy (IDENT) – Distribution-Shift Benchmark

Policy-or-Proxy provides generators, baselines, and evaluation scripts to study
sequence models under controlled distribution shift. Version 2.0 pivots entirely
to Wasserstein distance targets, with reusable YAML templates, multi-GPU
training, and richer evaluation (policy diagnostics + changepoints).

## 1. Quick Start
```
# Install editable package + optional plotting extras
git clone <repo>
cd policyProxy
pip install -e .[plot]
```

## 2. Data Generation (v2.0)

New configs live under `configs/base_v2.yaml` and `configs/ood/*.yaml`.
`base_v2.yaml` defines the IID split plus an `oid_templates` block that expands
to sixteen Wasserstein-targeted OOD datasets:

```
ood_state_w050,  ood_state_w100,  ood_state_w150,  ood_state_w200
ood_action_w050, ood_action_w100, ood_action_w150, ood_action_w200
ood_state_action_w050 ... w200
ood_policy_w050 ... w200
```

Generate everything (IID + OOD) in one shot:
```
python make_data.py --config configs/base_v2.yaml --out data/v2.0
```

`make_data.py` enhancements:
- YAML inheritance (`inherit:`) for lightweight overlays.
- `oid_templates` auto-expands to the full grid (with `oid_prefix: ood`).
- Pure Wasserstein tuning (`state`, `action`, `state_action`, `policy`).
- `--sweep` (optional) emits per-oid YAMLs for inspection without generating data.

Outputs include `config_used.yaml`, per-split indices, pilot tuning logs, and a
ready-to-use directory tree for evaluation.

## 3. Baselines

The action-prediction task now ships with a richer baseline set:
- `baselines/state_cond/train_cvae_pid.py` — CVAE conditioned on policy IDs (a-1).
- `baselines/state_cond/train_cvae.py` — GRU encoder/decoder CVAE; combine
  `--deterministic_latent` with `--global_latent/--no_global_latent` to toggle
  between the representation-focused (global) and stochastic variants.
- `baselines/state_cond/train_cvae_reg.py` — regularised CVAE with the same
  global/per-sample latent switch via `--no_global_latent`.
- `baselines/state_cond/train_cvae_transformer.py` — Transformer CVAE with
  sampled latent (no global `z`).
- `baselines/state_cond/train_gru.py` — GRU baseline.
- `baselines/state_cond/train_transformer_state.py` — deterministic Transformer
  conditioned on the state window (no VAE structure).

## 4. Evaluation Suite

Key scripts (all under `eval/`):
- `rollout.py` — rollouts on IID + all OOD splits, supports `--segmented_z` for
  deterministic CVAE segments and policy-ID conditioning.
- `diagnostics.py` — linear probe accuracy + clustering purity (`probe_accuracy`,
  `cluster_purity`, `rep_margin`).
- `rep_similarity.py` — margin trends across splits.
- `changepoints.py` — energy-based changepoint detection (reports `f1_tau3`,
  `mabe`, `delay_mean`).
- `wasserstein_report.py` — recomputes `ws_state`, `ws_action`, `ws_policy` for
  every split versus train.

`make_data.py` and `rollout.py` now consume `policy_id` labels exposed by
`tasks/common/dataset.py`, enabling policy-aware metrics everywhere.

## 5. Multi-GPU Workflow

`scripts/launch.sh` encapsulates the full pipeline:
1. Regenerate datasets from `configs/base_v2.yaml`.
2. Train four baselines on GPUs 0–3.
3. Evaluate rollouts, representation diagnostics, changepoints, and Wasserstein
   divergences for every split.
4. Write summaries to `runs/<model>/<TAG>/` and `reports/summary_<TAG>.md`.

Edit `RUN_TAG`, `DATA_ROOT`, or `CONFIG` as needed, then run:
```
bash scripts/launch.sh
```

## 6. Metrics at a Glance

Each run directory contains JSON sidecars with:
- `divergences_{split}.json` — `ws_state`, `ws_action`, `ws_policy` (targets ±0.02).
- `rollout_{split}.json` — `ADE`, `FDE`, `collision_rate`, `smoothness`.
- `rep_similarity.json`, `diagnostics.json` — representation robustness trends.
- `changepoints_{split}.json` — changepoint metrics.

## 7. Repository Map
```
configs/                Base + OOD YAMLs (inheritance-ready)
baselines/state_cond/   Training scripts (GRU, CVAE PID/REP, Transformer)
eval/                   Rollouts, diagnostics, changepoints, Wasserstein reports
scripts/                Launch helpers
tools/                  Distribution sampling + Wasserstein utilities
data/                  Generated datasets (v2.0+)
```

## 8. License & Citation
- Code: Apache 2.0 (`LICENSE-APACHE`)
- Data artifacts: CC BY 4.0 (`LICENSE-CC-BY`)
- Cite via `CITATION.cff`

For questions or bug reports, open an issue referencing the config/run and the
relevant `divergence_log.json`.
