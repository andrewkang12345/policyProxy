#!/bin/bash

# Policy-or-Proxy v5.0 Complete Workflow
# Features:
# 1. Gradient-based opponent optimization for state+action shifts
# 2. Direct configuration for policy shifts  
# 3. Proper divergence measurement (Wasserstein vs JS)
# 4. Separate plots with accurate units

set -e

# Configuration
RUN_TAG="v5_$(date +%Y%m%d_%H%M%S)"
DATA_ROOT="data/v5.0"
CONFIG="configs/base_v5.yaml"
REPORTS_DIR="reports/v5.0"
DEVICE="cpu"  # Change to "cuda" if GPU available
EPOCHS=20     # Reduced for demo

echo "🚀 Starting Policy-or-Proxy v5.0 Complete Workflow"
echo "Purpose: CVAE robustness with gradient-optimized opponents"
echo "Run Tag: $RUN_TAG"
echo "Data Root: $DATA_ROOT"
echo "Device: $DEVICE"
echo ""

# Create directories
mkdir -p $REPORTS_DIR/plots
mkdir -p $REPORTS_DIR/models
mkdir -p logs

# =============================================================================
# 1. DATA GENERATION WITH GRADIENT OPTIMIZATION
# =============================================================================
echo "📊 Step 1: V5.0 Data Generation"
echo "Key features:"
echo "  - Gradient-optimized opponents for state+action shifts"
echo "  - Direct config for policy shifts"
echo "  - Wasserstein targeting for state/action"
echo "  - JS targeting for policy"
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

echo "✅ V5.0 data generation completed"
echo ""

# =============================================================================
# 2. FIXED-Z CVAE TRAINING (Multiple Variants)
# =============================================================================
echo "🧠 Step 2: Training Fixed-Z CVAE variants"
echo "All variants use same architecture with fixed z per policy segment"
echo ""

# Define CVAE variants to train
CVAE_VARIANTS=("baseline" "policy_conditional" "learned_repr")

# Function to train a fixed-z CVAE variant
train_cvae_variant() {
    local variant=$1
    local save_dir="$REPORTS_DIR/models/cvae_fixed_z_${variant}_$RUN_TAG"
    
    echo "Training Fixed-Z CVAE variant: $variant"
    
    python baselines/state_cond/train_cvae_fixed_z.py \
        --data_root $DATA_ROOT \
        --save_dir $save_dir \
        --variant $variant \
        --epochs $EPOCHS \
        --batch_size 64 \
        --lr 1e-3 \
        --beta 1e-3 \
        --latent 16 \
        --hidden 128 \
        --device $DEVICE \
        2>&1 | tee logs/train_cvae_${variant}_$RUN_TAG.log
    
    if [ $? -eq 0 ]; then
        echo "✅ Fixed-Z CVAE $variant training completed"
        return 0
    else
        echo "❌ Fixed-Z CVAE $variant training failed"
        return 1
    fi
}

# Train all CVAE variants
successful_models=()
for variant in "${CVAE_VARIANTS[@]}"; do
    if train_cvae_variant "$variant"; then
        successful_models+=("cvae_fixed_z_${variant}_$RUN_TAG")
    fi
done

# Train GRU baseline for comparison
echo "Training GRU baseline..."
python baselines/state_cond/train_gru.py \
    --data_root $DATA_ROOT \
    --epochs $EPOCHS \
    --batch_size 64 \
    --lr 1e-3 \
    --device $DEVICE \
    --save_dir "$REPORTS_DIR/models/gru_baseline_$RUN_TAG" \
    --save_json "$REPORTS_DIR/models/gru_baseline_$RUN_TAG/results.json" \
    2>&1 | tee logs/train_gru_$RUN_TAG.log

if [ $? -eq 0 ]; then
    successful_models+=("gru_baseline_$RUN_TAG")
    echo "✅ GRU baseline training completed"
else
    echo "❌ GRU baseline training failed"
fi

echo "✅ Model training phase completed"
echo "Successfully trained models: ${successful_models[@]}"
echo ""

# =============================================================================
# 3. COMPREHENSIVE EVALUATION
# =============================================================================
echo "📊 Step 3: Comprehensive evaluation across all shifts"
echo "Evaluating on all generated splits with proper segmented z"
echo ""

# Get list of all splits
SPLITS=($(ls -d $DATA_ROOT/*/ | xargs -n 1 basename))
echo "Found splits: ${SPLITS[@]}"

# Function to evaluate a model on all splits
evaluate_model() {
    local model_dir=$1
    local model_name=$(basename "$model_dir")
    local model_path="$model_dir/model_best.pt"
    
    if [ ! -f "$model_path" ]; then
        echo "⚠️  Model not found: $model_path"
        return 1
    fi
    
    echo "Evaluating $model_name..."
    
    for split in "${SPLITS[@]}"; do
        if [[ "$split" == "optimized_opponents" || "$split" == "config_used.yaml" ]]; then
            continue  # Skip non-split directories
        fi
        
        echo "  Evaluating on $split..."
        
        # Rollout evaluation with segmented z for CVAE models
        if [[ "$model_name" == *"cvae"* ]]; then
            python eval/rollout.py \
                --data_root "$DATA_ROOT/$split" \
                --model "$model_path" \
                --episodes 20 \
                --device $DEVICE \
                --segmented_z \
                --save_json "$model_dir/rollout_${split}.json" \
                2>&1 | tee logs/eval_${model_name}_${split}_$RUN_TAG.log
        else
            # Regular evaluation for non-CVAE models
            python eval/rollout.py \
                --data_root "$DATA_ROOT/$split" \
                --model "$model_path" \
                --episodes 20 \
                --device $DEVICE \
                --save_json "$model_dir/rollout_${split}.json" \
                2>&1 | tee logs/eval_${model_name}_${split}_$RUN_TAG.log
        fi
    done
    
    echo "✅ $model_name evaluation completed"
}

# Evaluate all successful models
for model_name in "${successful_models[@]}"; do
    model_dir="$REPORTS_DIR/models/$model_name"
    evaluate_model "$model_dir"
done

echo "✅ Comprehensive evaluation completed"
echo ""

# =============================================================================
# 4. V5.0 ROBUSTNESS ANALYSIS
# =============================================================================
echo "🎨 Step 4: V5.0 robustness analysis and visualization"
echo "Creating separate plots with proper divergence units"
echo ""

# Copy results to expected location for plotting script
mkdir -p "$REPORTS_DIR/runs"
for model_name in "${successful_models[@]}"; do
    src_dir="$REPORTS_DIR/models/$model_name"
    dst_dir="$REPORTS_DIR/runs/$model_name"
    if [ -d "$src_dir" ]; then
        cp -r "$src_dir" "$dst_dir"
    fi
done

# Generate v5.0 plots
python scripts/analysis/create_detailed_performance_plots.py \
    --results_dir $DATA_ROOT \
    --output_dir $REPORTS_DIR/plots \
    2>&1 | tee logs/plotting_$RUN_TAG.log

echo "✅ V5.0 robustness analysis completed"
echo ""

# =============================================================================
# 5. RESULTS SUMMARY
# =============================================================================
echo "📈 Step 5: Results summary and validation"

# Count generated splits
num_state_shifts=$(ls -d $DATA_ROOT/ood_state_* 2>/dev/null | wc -l)
num_sa_shifts=$(ls -d $DATA_ROOT/ood_state_action_* 2>/dev/null | wc -l)
num_policy_shifts=$(ls -d $DATA_ROOT/ood_policy_* 2>/dev/null | wc -l)

echo ""
echo "🎉 Policy-or-Proxy v5.0 Workflow Complete!"
echo ""
echo "🔧 V5.0 Key Features Implemented:"
echo "  ✅ Gradient-based opponent optimization for state+action shifts"
echo "  ✅ Direct configuration for policy shifts"
echo "  ✅ Wasserstein distance measurement for state/action"
echo "  ✅ Jensen-Shannon divergence measurement for policy"
echo "  ✅ Fixed-Z CVAE architecture with segmented evaluation"
echo "  ✅ Separate plots with accurate divergence units"
echo ""
echo "📊 Generated Data:"
echo "  • IID splits: train, val, test"
echo "  • State shifts: $num_state_shifts (gradient-optimized)"
echo "  • State+action shifts: $num_sa_shifts (gradient-optimized)"
echo "  • Policy shifts: $num_policy_shifts (direct config)"
echo ""
echo "🧠 Trained Models:"
for model_name in "${successful_models[@]}"; do
    echo "  • $model_name"
done
echo ""
echo "📁 Generated Outputs:"
echo "  • Dataset: $DATA_ROOT"
echo "  • Models: $REPORTS_DIR/models/"
echo "  • Plots: $REPORTS_DIR/plots/"
echo "  • Logs: logs/"
echo ""
echo "📊 V5.0 Plots Generated:"
echo "  • State performance vs Wasserstein distance"
echo "  • Policy performance vs JS divergence"
echo "  • Divergence achievement validation"
echo "  • Model comparison summary"
echo ""
echo "🎯 Key Improvements Over v4.0:"
echo "  1. Gradient optimization enables precise state+action targeting"
echo "  2. Separate divergence measures for different shift types"
echo "  3. Achieved divergences reported (not targets)"
echo "  4. Individual plots for each analysis type"
echo ""
echo "Run completed at: $(date)"
echo "🚀 v5.0 workflow provides definitive CVAE robustness benchmarking!"
