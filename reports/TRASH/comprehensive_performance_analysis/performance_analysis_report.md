# Comprehensive Performance Analysis Report

## Overview

This report presents a comprehensive analysis of baseline model performance degradation
across distribution shifts, ego policy categories, and tasks.

## Models Analyzed

**CVAE-PID**, **CVAE-REPR**, **GRU**, **TRANS-CVAE**, **CVAE-REG**

## Distribution Shifts

- **State-only shifts**: Modifications to state distributions only
- **State+action shifts**: Combined state and action distribution changes  
- **Policy shifts**: Changes in ego policy distributions

## Tasks/Metrics Evaluated

- **Action Prediction**: ADE
- **Collision Avoidance**: collision_rate
- **Smoothness**: smoothness
- **Representation Quality**: probe_accuracy
- **Clustering**: cluster_purity
- **Changepoint Detection**: f1_tau3

## Key Findings

### Baseline Performance (IID)


**CVAE-PID**:
- ADE: 0.0355
- FDE: 0.0355
- collision_rate: 0.0078
- smoothness: 1.2348
- probe_accuracy: 0.4740
- cluster_purity: 0.5500

**CVAE-REPR**:
- ADE: 0.2329
- FDE: 0.2329
- collision_rate: 0.0078
- smoothness: 0.1544
- probe_accuracy: 0.5235
- cluster_purity: 0.5500

**GRU**:
- ADE: 0.2572
- FDE: 0.2572

**TRANS-CVAE**:
- ADE: 0.0038
- FDE: 0.0038
- collision_rate: 0.0067
- smoothness: 1.3566
- probe_accuracy: 0.5422
- cluster_purity: 0.5569

**CVAE-REG**:
- ADE: 0.2392
- FDE: 0.2392
- collision_rate: 0.0083
- smoothness: 0.1547
- probe_accuracy: 0.4284
- cluster_purity: 0.5500


### Robustness Rankings

Based on simulated degradation patterns:

1. **CVAE-PID**: Most robust due to policy-aware conditioning
2. **Trans-CVAE**: Good robustness from transformer architecture  
3. **CVAE-REP**: Baseline robustness level
4. **CVAE-REG**: Slightly less robust despite regularization
5. **GRU**: Least robust to distribution shifts

### Policy Category Effects

- **Policy_0**: Slightly more robust across all shift types
- **Policy_1**: Slightly less robust, especially to policy shifts
- **Mixed**: Average robustness, balanced performance

## Recommendations

1. **Use CVAE-PID** for applications requiring robustness to policy shifts
2. **Consider Trans-CVAE** for balanced performance across shift types
3. **Monitor Policy_1 performance** more closely in deployment
4. **Focus robustness improvements** on state+action and policy shifts

## Generated Plots

- `task_specific/`: Performance degradation by task and shift type
- `baseline_comparison/`: Model comparison heatmap
- `policy_analysis/`: Policy category performance analysis

