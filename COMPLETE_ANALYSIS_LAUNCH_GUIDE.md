# Complete Analysis Launch Guide

## Overview

The **`launch_complete_analysis.sh`** script provides a single command to run the entire Policy-or-Proxy analysis pipeline from start to finish, generating all results needed for comprehensive performance analysis.

## What It Does

### 🚀 **Complete End-to-End Pipeline**
```bash
bash scripts/workflow/launch_complete_analysis.sh
```

This script executes **7 major steps** in sequence:

### **Step 1: Data Generation**
- Generates datasets with three distribution shifts
- Uses gradient-based opponent optimization
- Creates state-only, state+action, and policy shifts
- Applies proper Wasserstein and Jensen-Shannon targeting

### **Step 2: Model Training**  
- Trains **5 baseline models**:
  - CVAE-PID (Policy ID conditional)
  - CVAE-REPR (Representation learning)
  - CVAE-REG (Regularized)
  - GRU (Recurrent baseline)
  - TRANS-CVAE (Transformer-based)

### **Step 3: Model Evaluation**
- Evaluates all models on all distribution shifts
- Generates rollout metrics (ADE, collision rate, smoothness)
- Performs representation similarity analysis
- Creates diagnostic reports

### **Step 4: Task B Analysis**
- Runs policy classification
- Performs changepoint detection
- Tests representation quality

### **Step 5: Detailed Performance Analysis**
- Generates **36 separate plots** with clear units
- Creates degradation curves for each task/shift/policy
- Produces task comparison plots
- Builds robustness ranking heatmap

### **Step 6: Three-Shifts Specialized Analysis**
- Focuses on gradient-optimized distribution shifts
- Analyzes performance vs actual achieved divergences
- Provides policy category analysis
- Creates specialized robustness metrics

### **Step 7: Summary & Validation**
- Generates comprehensive summary report
- Validates all outputs
- Provides clear results location guide

## Expected Outputs

### **Generated Files**
```
reports/complete_analysis_YYYYMMDD_HHMMSS/
├── COMPLETE_ANALYSIS_SUMMARY.md      # Main summary report
├── models/                           # All trained models
│   ├── cvae_pid_TIMESTAMP/
│   ├── cvae_repr_TIMESTAMP/
│   ├── cvae_reg_TIMESTAMP/
│   ├── gru_TIMESTAMP/
│   └── trans_cvae_TIMESTAMP/
├── evaluation/                       # Model evaluation results
│   ├── rollout_results/
│   ├── similarity_analysis/
│   └── diagnostics/
├── task_b/                          # Task B analysis results
│   ├── policy_classification/
│   └── changepoint_detection/
└── analysis/                        # Performance analysis plots
    ├── detailed_performance/        # 36 detailed plots
    │   ├── degradation_curves/      # 30 degradation curves
    │   ├── task_comparisons/        # 5 task comparisons
    │   └── robustness_ranking/      # 1 robustness heatmap
    └── three_shifts/               # Specialized analysis
        ├── performance_plots/
        ├── policy_analysis/
        └── robustness_metrics/
```

### **Key Plots (36 Total)**
1. **Degradation Curves (30 plots)**: Performance vs shift strength
   - 5 tasks × 3 shift types × 2 policy categories
   - Separate lines for each CVAE variant
   - Clear axis units (Wasserstein distance / JS divergence)

2. **Task Comparisons (5 plots)**: Cross-shift analysis
   - Action prediction, collision avoidance, smoothness
   - Representation quality, policy clustering
   - Multiple baselines per plot

3. **Robustness Ranking (1 plot)**: Overall performance
   - Heatmap of baseline robustness
   - Across all tasks and shift types

## Configuration

### **Default Settings**
- **Device**: CPU (change to "cuda" in script for GPU)
- **Epochs**: 50 (full training)
- **Batch Size**: 64
- **Data Config**: `configs/base_v5.yaml`

### **Customization**
Edit the script variables at the top:
```bash
DEVICE="cuda"        # Use GPU if available
EPOCHS=100           # More training epochs
BATCH_SIZE=128       # Larger batch size
```

## Runtime Expectations

### **Typical Runtime** (CPU)
- **Data Generation**: 30-60 minutes
- **Model Training**: 2-4 hours (5 models × 50 epochs)
- **Evaluation**: 30-60 minutes
- **Analysis**: 15-30 minutes
- **Total**: ~4-6 hours

### **With GPU**
- **Model Training**: 30-60 minutes
- **Total**: ~2-3 hours

## Monitoring Progress

### **Live Monitoring**
```bash
# Watch overall progress
tail -f logs/*_complete_*.log

# Monitor specific training
tail -f logs/train_cvae_pid_complete_*.log

# Check analysis progress
tail -f logs/detailed_analysis_complete_*.log
```

### **Check Intermediate Results**
```bash
# See what's been generated
ls reports/complete_analysis_*/

# Check model training progress
ls reports/complete_analysis_*/models/

# View completed plots
find reports/complete_analysis_*/analysis -name "*.png" | wc -l
```

## Success Criteria

### **Full Success**
- ✅ All 5 models trained successfully
- ✅ 36 performance plots generated
- ✅ Task B analysis completed
- ✅ Summary report created

### **Partial Success**
- ✅ At least 3 models trained
- ✅ Core analysis plots generated
- ⚠️ Some components may need attention (check logs)

## Troubleshooting

### **Common Issues**
1. **Memory Errors**: Reduce batch size or use GPU
2. **Training Failures**: Check data generation logs first
3. **Plot Generation Issues**: Verify evaluation results exist
4. **Permission Errors**: Ensure script is executable

### **Quick Fixes**
```bash
# Make script executable
chmod +x scripts/workflow/launch_complete_analysis.sh

# Check disk space
df -h

# Verify dependencies
pip install -e .[plot]
```

## Alternative Launch Options

### **If You Need Faster Results**
```bash
# Core workflow only (basic training)
bash scripts/workflow/launch_v5_workflow.sh

# Just generate plots from existing data
python scripts/analysis/create_detailed_performance_plots.py
```

### **If You Need Specific Analysis**
```bash
# Task framework only
bash scripts/workflow/launch_v5_proper_tasks.sh

# Three-shifts analysis only
python scripts/analysis/create_three_shifts_performance_analysis.py
```

## Summary

The `launch_complete_analysis.sh` script is your **one-command solution** for complete Policy-or-Proxy performance analysis. It handles everything from data generation to final plots, producing publication-ready results with clear documentation and organized outputs.

**Just run**: `bash scripts/workflow/launch_complete_analysis.sh` and wait for complete results!
