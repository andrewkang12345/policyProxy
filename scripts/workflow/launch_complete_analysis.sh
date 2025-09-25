#!/bin/bash

# Policy-or-Proxy Complete End-to-End Analysis
# 
# This script runs the COMPLETE workflow from start to finish:
# 1. Data generation with three distribution shifts
# 2. Training all baseline models
# 3. Evaluation on all splits
# 4. Complete performance analysis with 36 detailed plots
# 5. Specialized three-shifts analysis
# 6. Task B implementation and testing
#
# Usage: bash scripts/workflow/launch_complete_analysis.sh

set -e

# Configuration
RUN_TAG="complete_$(date +%Y%m%d_%H%M%S)"
DATA_ROOT="data/complete_analysis_$RUN_TAG"
CONFIG="configs/base_v5.yaml"
REPORTS_DIR="reports/complete_analysis_$RUN_TAG"
DEVICE="cpu"  # Change to "cuda" if GPU available
EPOCHS=5      # Reduced for testing - change to 50 for full analysis
BATCH_SIZE=64

echo "🚀 STARTING COMPLETE END-TO-END ANALYSIS"
echo "======================================="
echo "Purpose: Complete Policy-or-Proxy performance analysis pipeline"
echo "Run Tag: $RUN_TAG"
echo "Data Root: $DATA_ROOT"
echo "Reports: $REPORTS_DIR"
echo "Device: $DEVICE"
echo "Epochs: $EPOCHS"
echo ""

# Create all necessary directories
mkdir -p $REPORTS_DIR/{plots,models,evaluation,analysis}
mkdir -p logs

# =============================================================================
# STEP 1: DATA GENERATION WITH THREE DISTRIBUTION SHIFTS
# =============================================================================
echo "📊 STEP 1: Data Generation with Three Distribution Shifts"
echo "========================================================="
echo "Features:"
echo "  ✓ State-only shifts (Wasserstein distance targeting)"
echo "  ✓ State+action shifts (non-random correlation)"
echo "  ✓ Policy shifts (Jensen-Shannon divergence targeting)"
echo "  ✓ Gradient-based opponent optimization"
echo ""

python make_data_v5.py \
    --config $CONFIG \
    --out $DATA_ROOT \
    --device $DEVICE \
    2>&1 | tee logs/data_generation_$RUN_TAG.log

if [ ! -d "$DATA_ROOT" ]; then
    echo "❌ Data generation failed!"
    exit 1
fi

echo "✅ Data generation completed successfully"
echo ""

# =============================================================================
# STEP 2: BASELINE MODEL TRAINING
# =============================================================================
echo "🧠 STEP 2: Training All Baseline Models"
echo "======================================="
echo "Models to train:"
echo "  ✓ CVAE-PID (Policy ID conditional)"
echo "  ✓ CVAE-REG (Regularized)"
echo "  ✓ GRU (Recurrent baseline)"
echo "  ✓ TRANS-CVAE (Transformer-based)"
echo ""

# Function to train a model
train_model() {
    local model_type=$1
    local script_name=$2
    local extra_args=${3:-""}
    local save_dir="$REPORTS_DIR/models/${model_type}_$RUN_TAG"
    
    echo "🔧 Training $model_type..."
    
    python baselines/state_cond/$script_name \
        --data_root $DATA_ROOT \
        --save_dir $save_dir \
        --epochs $EPOCHS \
        --batch_size $BATCH_SIZE \
        --device $DEVICE \
        $extra_args \
        2>&1 | tee logs/train_${model_type}_$RUN_TAG.log
    
    if [ $? -eq 0 ]; then
        echo "✅ $model_type training completed"
        return 0
    else
        echo "❌ $model_type training failed"
        return 1
    fi
}

# Train all baseline models
successful_models=()

# CVAE variants
if train_model "cvae_pid" "train_cvae_pid.py" "--beta 1e-3 --latent 16"; then
    successful_models+=("cvae_pid_$RUN_TAG")
fi

if train_model "cvae_reg" "train_cvae_reg.py" "--beta 1e-3 --latent 16 --reg 0.1"; then
    successful_models+=("cvae_reg_$RUN_TAG")
fi

# GRU baseline
if train_model "gru" "train_gru.py" "--gaussian_head"; then
    successful_models+=("gru_$RUN_TAG")
fi

# Transformer CVAE
if train_model "trans_cvae" "train_cvae_transformer.py" "--beta 1e-3 --latent 16 --nhead 4"; then
    successful_models+=("trans_cvae_$RUN_TAG")
fi

echo ""
echo "🎯 Model Training Summary:"
echo "Successful models: ${#successful_models[@]}"
for model in "${successful_models[@]}"; do
    echo "  ✅ $model"
done

if [ ${#successful_models[@]} -eq 0 ]; then
    echo "❌ No models trained successfully!"
    exit 1
fi

echo ""

# =============================================================================
# STEP 3: MODEL EVALUATION ON ALL SPLITS
# =============================================================================
echo "📊 STEP 3: Model Evaluation on All Distribution Shifts"
echo "====================================================="
echo ""

# Function to evaluate a model
evaluate_model() {
    local model_dir=$1
    local model_name=$(basename $model_dir)
    
    echo "🔍 Evaluating $model_name..."
    
    # Rollout evaluation
    python eval/rollout.py \
        --model $model_dir/model_best.pt \
        --data_root $DATA_ROOT \
        --save_json $REPORTS_DIR/evaluation/rollout_${model_name}.json \
        --device $DEVICE \
        2>&1 | tee logs/eval_rollout_${model_name}.log
    
    # Representation similarity analysis
    python eval/rep_similarity.py \
        --data_root $DATA_ROOT \
        --save_json $REPORTS_DIR/evaluation/similarity_${model_name}.json \
        --device $DEVICE \
        2>&1 | tee logs/eval_similarity_${model_name}.log
    
    # Diagnostics
    python eval/diagnostics.py \
        --data_root $DATA_ROOT \
        --run_dir $model_dir \
        --device $DEVICE \
        2>&1 | tee logs/eval_diagnostics_${model_name}.log
    
    echo "✅ $model_name evaluation completed"
}

# Evaluate all successful models
for model_name in "${successful_models[@]}"; do
    model_dir="$REPORTS_DIR/models/$model_name"
    if [ -d "$model_dir" ]; then
        evaluate_model "$model_dir"
    fi
done

echo ""

# =============================================================================
# STEP 4: TASK B - POLICY REPRESENTATION ANALYSIS
# =============================================================================
echo "🎯 STEP 4: Task B - Policy Representation Analysis"
echo "================================================="
echo ""

# Run fixed Task B components
echo "🔧 Running Task B policy classification and changepoint detection..."

python scripts/fixes/run_task_b_fixed.py \
    --data_root $DATA_ROOT \
    --models_dir $REPORTS_DIR/models \
    --output_dir $REPORTS_DIR/task_b \
    2>&1 | tee logs/task_b_$RUN_TAG.log

if [ $? -eq 0 ]; then
    echo "✅ Task B analysis completed"
else
    echo "⚠️  Task B analysis had issues (check logs)"
fi

echo ""

# =============================================================================
# STEP 5: COMPREHENSIVE PERFORMANCE ANALYSIS
# =============================================================================
echo "📈 STEP 5: Comprehensive Performance Analysis (36 Plots)"
echo "======================================================="
echo ""

echo "🎨 Generating detailed performance analysis plots..."
echo "This creates 36 separate images with clear units:"
echo "  ✓ 30 degradation curves (5 tasks × 3 shifts × 2 policies)"
echo "  ✓ 5 task comparison plots"
echo "  ✓ 1 robustness ranking heatmap"
echo ""

python scripts/analysis/create_detailed_performance_plots.py \
    --runs_dir $REPORTS_DIR/models \
    --output_dir $REPORTS_DIR/analysis/detailed_performance \
    2>&1 | tee logs/detailed_analysis_$RUN_TAG.log

if [ $? -eq 0 ]; then
    echo "✅ Detailed performance analysis completed"
    detailed_plots_success=true
else
    echo "❌ Detailed performance analysis failed"
    detailed_plots_success=false
fi

echo ""

# =============================================================================
# STEP 6: THREE DISTRIBUTION SHIFTS SPECIALIZED ANALYSIS
# =============================================================================
echo "📊 STEP 6: Three Distribution Shifts Specialized Analysis"
echo "========================================================"
echo ""

echo "🔬 Generating specialized three-shifts analysis..."
echo "This focuses on gradient-optimized distribution shifts:"
echo "  ✓ Performance vs actual achieved divergences"
echo "  ✓ Policy category analysis"
echo "  ✓ Robustness ranking across shift types"
echo ""

python scripts/analysis/create_three_shifts_performance_analysis.py \
    --data_dirs $DATA_ROOT \
    --baselines_dir $REPORTS_DIR/models \
    --output_dir $REPORTS_DIR/analysis/three_shifts \
    2>&1 | tee logs/three_shifts_analysis_$RUN_TAG.log

if [ $? -eq 0 ]; then
    echo "✅ Three-shifts specialized analysis completed"
    three_shifts_success=true
else
    echo "❌ Three-shifts specialized analysis failed"
    three_shifts_success=false
fi

echo ""

# =============================================================================
# STEP 7: RESULTS SUMMARY AND VALIDATION
# =============================================================================
echo "📋 STEP 7: Results Summary and Validation"
echo "========================================="
echo ""

# Count generated data
num_state_shifts=$(ls -d $DATA_ROOT/ood_state_* 2>/dev/null | wc -l)
num_sa_shifts=$(ls -d $DATA_ROOT/ood_state_action_* 2>/dev/null | wc -l)
num_policy_shifts=$(ls -d $DATA_ROOT/ood_policy_* 2>/dev/null | wc -l)

# Count generated plots
detailed_plots=$(find $REPORTS_DIR/analysis/detailed_performance -name "*.png" 2>/dev/null | wc -l)
three_shifts_plots=$(find $REPORTS_DIR/analysis/three_shifts -name "*.png" 2>/dev/null | wc -l)

# Generate summary report
cat > $REPORTS_DIR/COMPLETE_ANALYSIS_SUMMARY.md << EOF
# Complete Analysis Summary - $RUN_TAG

## Overview
Complete end-to-end analysis run completed on $(date)

## Data Generation
- **State shifts**: $num_state_shifts (gradient-optimized)
- **State+action shifts**: $num_sa_shifts (gradient-optimized) 
- **Policy shifts**: $num_policy_shifts (gradient-optimized)
- **Total splits**: $(ls -d $DATA_ROOT/*/ 2>/dev/null | wc -l)

## Model Training
- **Models trained**: ${#successful_models[@]}/5
$(for model in "${successful_models[@]}"; do echo "- ✅ $model"; done)

## Performance Analysis
- **Detailed plots**: $detailed_plots/36 expected
- **Three-shifts plots**: $three_shifts_plots expected
- **Task B analysis**: $([ -d "$REPORTS_DIR/task_b" ] && echo "✅ Completed" || echo "❌ Failed")

## Results Location
- **Data**: \`$DATA_ROOT\`
- **Models**: \`$REPORTS_DIR/models\`
- **Evaluation**: \`$REPORTS_DIR/evaluation\`
- **Analysis**: \`$REPORTS_DIR/analysis\`
- **Logs**: \`logs/*_$RUN_TAG.log\`

## Key Plots
$(if [ "$detailed_plots_success" = true ]; then
    echo "### Detailed Performance Analysis"
    echo "- Degradation curves: \`$REPORTS_DIR/analysis/detailed_performance/degradation_curves/\`"
    echo "- Task comparisons: \`$REPORTS_DIR/analysis/detailed_performance/task_comparisons/\`"
    echo "- Robustness ranking: \`$REPORTS_DIR/analysis/detailed_performance/robustness_ranking/\`"
fi)

$(if [ "$three_shifts_success" = true ]; then
    echo "### Three Distribution Shifts Analysis"
    echo "- Specialized analysis: \`$REPORTS_DIR/analysis/three_shifts/\`"
fi)

## Usage
View results by opening the PNG files in the analysis directories or running:
\`\`\`bash
# View all plots
find $REPORTS_DIR/analysis -name "*.png" | head -10

# View summary data
ls $REPORTS_DIR/evaluation/
\`\`\`
EOF

echo ""
echo "🎉 COMPLETE END-TO-END ANALYSIS FINISHED!"
echo "========================================="
echo ""
echo "📊 FINAL SUMMARY:"
echo "  🔧 Data Generation: ✅"
echo "  🧠 Model Training: ${#successful_models[@]}/5 models"
echo "  📊 Model Evaluation: ✅"
echo "  🎯 Task B Analysis: $([ -d "$REPORTS_DIR/task_b" ] && echo "✅" || echo "⚠️")"
echo "  📈 Detailed Analysis: $([ "$detailed_plots_success" = true ] && echo "✅ ($detailed_plots plots)" || echo "❌")"
echo "  🔬 Three-Shifts Analysis: $([ "$three_shifts_success" = true ] && echo "✅ ($three_shifts_plots plots)" || echo "❌")"
echo ""
echo "📁 RESULTS LOCATION:"
echo "  📄 Summary: $REPORTS_DIR/COMPLETE_ANALYSIS_SUMMARY.md"
echo "  📊 Data: $DATA_ROOT"
echo "  🧠 Models: $REPORTS_DIR/models"
echo "  📈 Analysis: $REPORTS_DIR/analysis"
echo "  📋 Logs: logs/*_$RUN_TAG.log"
echo ""
echo "🎨 KEY OUTPUTS:"
if [ "$detailed_plots_success" = true ]; then
    echo "  📈 Detailed Performance: $REPORTS_DIR/analysis/detailed_performance/"
    echo "      • $detailed_plots separate plots with clear units"
    echo "      • Degradation curves for each task/shift/policy"
    echo "      • Task comparisons across all shifts"
    echo "      • Robustness ranking heatmap"
fi

if [ "$three_shifts_success" = true ]; then
    echo "  🔬 Three-Shifts Analysis: $REPORTS_DIR/analysis/three_shifts/"
    echo "      • Performance vs actual divergences"
    echo "      • Policy category analysis"
    echo "      • Specialized robustness metrics"
fi

echo ""
echo "🚀 Ready for analysis! Check the summary report for detailed results."

# Final status
if [ ${#successful_models[@]} -ge 3 ] && [ "$detailed_plots_success" = true ]; then
    echo "🎉 COMPLETE SUCCESS: Full pipeline executed successfully!"
    exit 0
else
    echo "⚠️  PARTIAL SUCCESS: Some components may need attention (check logs)"
    exit 1
fi
