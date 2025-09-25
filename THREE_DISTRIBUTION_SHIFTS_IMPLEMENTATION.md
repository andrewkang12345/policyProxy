# Three Distribution Shifts Implementation Summary

## Overview

This implementation provides three different distribution shifts with gradient-based optimization for opponent movement, as requested:

1. **State-only shifts** - Gradient-optimized opponents that modify state distributions
2. **State+action shifts** - Gradient-optimized opponents with non-random ego policy correlation to state  
3. **Policy shifts** - Gradient-optimized opponents targeting policy distribution changes

All three shifts use gradient-based optimization for opponent movement, creating controlled distribution shifts for benchmarking sequence models under distribution shift.

## Key Implementation Components

### 1. Enhanced Differentiable Opponent (`DifferentiableOpponent`)

**Location**: `/mnt/data/policyProxy/generator/gradient_opponent_optimizer.py`

The core differentiable opponent model supports all three shift types:

- **State-only mode**: Focuses on modifying state distributions via position bias
- **State+action mode**: Creates structured state-action correlation using learnable correlation matrix  
- **Policy mode**: Responds differently based on ego policy IDs to create policy distribution shifts

**Key features**:
- Parameterized state and action biases
- State correlation matrix for structured state-action dependencies
- Policy preference parameters for policy-dependent responses
- Gradient-friendly forward pass with no in-place operations

### 2. Gradient-Based Optimization Engine (`GradientOpponentOptimizer`)

**Location**: `/mnt/data/policyProxy/generator/gradient_opponent_optimizer.py`

Optimizes opponent policies using gradient descent to achieve target divergences:

- **Wasserstein distance** computation for state and action shifts
- **Jensen-Shannon divergence** computation for policy shifts  
- Dynamic episode generation with opponent influence on state trajectories
- Convergence tracking and early stopping

**Key methods**:
- `optimize_opponent()`: Main optimization loop for single target
- `generate_episode_data()`: Creates episodes with opponent-influenced dynamics
- `compute_wasserstein_distance()`: Differentiable Wasserstein distance
- `compute_js_divergence()`: Jensen-Shannon divergence for policy shifts

### 3. Enhanced Data Generation Pipeline (`make_data_v5.py`)

**Location**: `/mnt/data/policyProxy/make_data_v5.py`

Orchestrates the complete data generation workflow:

1. Generate baseline IID splits
2. Load baseline data for optimization targets
3. Optimize opponents for each shift type and target divergence
4. Generate shifted datasets using optimized opponents
5. Compute and save divergence metrics

**Key features**:
- Support for all three shift types in single configuration
- Parallel optimization of multiple targets
- Divergence validation and reporting
- Model persistence for optimized opponents

### 4. Updated Configuration

**Location**: `/mnt/data/policyProxy/configs/base_v5.yaml`

Enhanced configuration supporting all three shift types:

```yaml
gradient_optimization:
  enabled: true
  targets:
    # State-only shifts
    - shift_kind: "state_only"
      target_divergence: 0.05
      
    # State+action shifts  
    - shift_kind: "state_action"
      target_divergence: 0.05
      
    # Policy shifts
    - shift_kind: "policy"
      target_divergence: 0.05
```

## Technical Implementation Details

### State-Only Shifts

- **Mechanism**: Opponent modifies ego state distribution via learnable position and velocity biases
- **Gradient flow**: State trajectories → Position bias → Wasserstein distance → Loss
- **Metric**: Wasserstein distance on state positions
- **Non-random correlation**: Weak coupling between state changes and opponent actions

### State+Action Shifts

- **Mechanism**: Structured state-action correlation via learnable correlation matrix
- **Gradient flow**: State trajectories → Correlation matrix → Combined state+action divergence → Loss  
- **Metric**: Average of state and action Wasserstein distances
- **Non-random correlation**: Explicit correlation matrix creates deterministic state-action dependencies

### Policy Shifts

- **Mechanism**: Policy-dependent opponent response via learnable policy preferences
- **Gradient flow**: Policy IDs → Policy preferences → Policy distribution → JS divergence → Loss
- **Metric**: Jensen-Shannon divergence on policy distributions
- **Non-random correlation**: Opponent responds systematically differently to different ego policies

## Verification Results

Testing with `/mnt/data/policyProxy/configs/test_v5_three_shifts.yaml` shows:

✅ **State-only shifts**: Successfully optimized opponents for targets 0.05 and 0.10
✅ **State+action shifts**: Successfully optimized opponents for targets 0.05 and 0.10  
✅ **Policy shifts**: Successfully optimized opponents for targets 0.05 and 0.10

**Generated artifacts**:
- 6 optimized opponent models (2 per shift type)
- 6 shifted datasets with corresponding episodes
- Convergence histories and divergence metrics
- Configuration files for reproducibility

## Usage Example

```bash
# Generate all three distribution shifts
python make_data_v5.py \
    --config configs/base_v5.yaml \
    --out data/v5_three_shifts \
    --device cpu

# Generated structure:
# data/v5_three_shifts/
# ├── train/              # IID baseline
# ├── val/                # IID baseline  
# ├── test/               # IID baseline
# ├── ood_state_only_*/   # State-only shifts
# ├── ood_state_action_*/ # State+action shifts
# ├── ood_policy_*/       # Policy shifts
# ├── optimized_opponents/ # Trained opponent models
# └── v5_divergences.json # Metrics summary
```

## Key Advantages

1. **Unified framework**: All three shift types use the same gradient-based optimization approach
2. **Controlled shifts**: Precise targeting of specific divergence levels  
3. **Non-random correlations**: Structured state-action and policy dependencies
4. **Reproducible**: Saved models and configurations enable exact replication
5. **Extensible**: Easy to add new shift types or modify existing ones
6. **Efficient**: Gradient-based optimization converges faster than search-based methods

## Future Extensions

- **Multi-objective optimization**: Simultaneously optimize multiple divergence types
- **Temporal shifts**: Optimize opponents for temporal distribution changes
- **Hierarchical shifts**: Nested state/action/policy shift combinations
- **Adaptive opponents**: Dynamic opponent adjustment during episodes
- **Robustness evaluation**: Test model performance across shift gradients

This implementation provides a comprehensive foundation for studying sequence model robustness under controlled distribution shifts with gradient-optimized opponent behaviors.
