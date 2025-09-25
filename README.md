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

## 2. Data Generation (v5.0)

Current configs use `configs/base_v5.yaml` for the three distribution shifts implementation:

**Three Distribution Shifts (Gradient-Optimized)**:
```
state_only_*     - State distribution shifts with gradient-optimized opponents
state_action_*   - State+action shifts with non-random correlation
policy_*         - Policy distribution shifts with gradient-based optimization
```

Generate everything (IID + all three shift types):
```
python make_data_v5.py --config configs/base_v5.yaml --out data/v5.0
```

**Key Features**:
- **Gradient-based opponent optimization** for all shift types
- **Wasserstein targeting** for state/action shifts  
- **Jensen-Shannon targeting** for policy shifts
- **Non-random correlations** between state and actions
- **Policy-aware** opponent responses

Outputs include optimized opponent models, divergence metrics, and evaluation-ready datasets.

## 3. Baselines

Four training scripts cover the main study axes:
- `baselines/state_cond/train_cvae_pid.py` — CVAE conditioned on policy IDs (a-1).
- `baselines/state_cond/train_cvae.py` (`--deterministic_latent`) — representation-aware CVAE (a-2).
- `baselines/state_cond/train_gru.py` — GRU baseline.
- `baselines/state_cond/train_cvae_transformer.py` — Transformer CVAE.

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

## 5. Complete Workflow

### **End-to-End Analysis** (Recommended)
`scripts/workflow/launch_complete_analysis.sh` runs the COMPLETE pipeline from start to finish:
1. Generate datasets with three distribution shifts
2. Train all 5 baseline models (CVAE variants + GRU + Transformer)
3. Evaluate models on all splits with comprehensive metrics
4. Run Task B policy representation analysis
5. Generate 36 detailed performance plots with clear units
6. Create specialized three-shifts analysis
7. Produce complete summary report

```
bash scripts/workflow/launch_complete_analysis.sh
```

### **Individual Components**
For running specific parts:

**Core workflow** (data + basic training):
```
bash scripts/workflow/launch_v5_workflow.sh
```

**Detailed performance analysis** (36 separate images):
```
python scripts/analysis/create_detailed_performance_plots.py
```

**Three-shifts specialized analysis**:
```
python scripts/analysis/create_three_shifts_performance_analysis.py
```

## 6. Metrics at a Glance

Each run directory contains JSON sidecars with:
- `divergences_{split}.json` — `ws_state`, `ws_action`, `ws_policy` (targets ±0.02).
- `rollout_{split}.json` — `ADE`, `FDE`, `collision_rate`, `smoothness`.
- `rep_similarity.json`, `diagnostics.json` — representation robustness trends.
- `changepoints_{split}.json` — changepoint metrics.

## 7. Repository Map
```
configs/                V5.0 configurations (three distribution shifts)
baselines/state_cond/   Training scripts (GRU, CVAE PID/REP, Transformer)
eval/                   Rollouts, diagnostics, changepoints, Wasserstein reports
scripts/                Organized workflow and analysis scripts
  ├── analysis/         Performance analysis and plotting
  │   ├── create_detailed_performance_plots.py  (36 separate plots)
  │   └── create_three_shifts_performance_analysis.py  (specialized)
  ├── workflow/         Main workflow launchers
  │   ├── launch_complete_analysis.sh  (END-TO-END: full pipeline)
  │   ├── launch_v5_workflow.sh  (core workflow)
  │   └── launch_v5_proper_tasks.sh  (task framework)
  ├── task_framework/   Task A/B specific components
  │   └── create_proper_task_plots.py
  ├── fixes/            Bug fixes and working implementations
  │   ├── fix_task_b_issues.py
  │   └── run_task_b_fixed.py
  └── archived/         Historical scripts for reference
generator/              Gradient-based opponent optimization
tools/                  Distribution sampling + Wasserstein utilities
data/                   Current v5.0 datasets (three shifts)
docs/                   Documentation and archived materials
runs/                   Current v4.0 baseline models only
reports/                Current analysis results only
```

## 8. License & Citation
- Code: Apache 2.0 (`LICENSE-APACHE`)
- Data artifacts: CC BY 4.0 (`LICENSE-CC-BY`)
- Cite via `CITATION.cff`

For questions or bug reports, open an issue referencing the config/run and the
relevant `divergence_log.json`.
