#!/bin/bash

# Policy-or-Proxy v5.0 Proper Task Framework
# Implements the correct experimental structure with:
# Task A: Action Output (2 subtasks, both segment-aware)
# Task B: Policy Representation (2 subtasks, mixed segment awareness)

set -e

# Configuration
RUN_TAG="v5_proper_$(date +%Y%m%d_%H%M%S)"
DATA_ROOT="data/v5.0_simple"  # Use existing v5.0 dataset
REPORTS_DIR="reports/v5.0_proper"
DEVICE="cpu"
EPOCHS=10     # More epochs for proper evaluation

echo "🚀 Starting Policy-or-Proxy v5.0 Proper Task Framework"
echo "Purpose: Correct experimental structure with proper task definitions"
echo "Run Tag: $RUN_TAG"
echo "Data Root: $DATA_ROOT"
echo ""

# Create directories
mkdir -p $REPORTS_DIR/task_a_action_output
mkdir -p $REPORTS_DIR/task_b_policy_representation
mkdir -p $REPORTS_DIR/plots
mkdir -p logs

# Verify dataset exists
if [ ! -d "$DATA_ROOT" ]; then
    echo "❌ Dataset not found at $DATA_ROOT. Run v5.0 data generation first."
    exit 1
fi

echo "📊 Using existing v5.0 dataset: $DATA_ROOT"
echo ""

# =============================================================================
# TASK A: ACTION OUTPUT (Both subtasks aware of fixed policy segments)
# =============================================================================
echo "🎯 TASK A: ACTION OUTPUT (Segment-Aware Evaluation)"
echo "Both subtasks know the fixed policy segments and use this information"
echo ""

# Task A1: Ground Truth Policy as z
echo "📋 Task A1: Ground Truth Policy ID as Latent z"
echo "  - Model: Fixed-Z CVAE with policy ID embedding"
echo "  - Latent: z = policy_embedding[gt_policy_id]"
echo "  - Segments: Known (segment-aware evaluation)"
echo ""

python baselines/state_cond/train_cvae_fixed_z.py \
    --data_root $DATA_ROOT \
    --save_dir "$REPORTS_DIR/task_a_action_output/a1_gt_policy_as_z_$RUN_TAG" \
    --variant policy_conditional \
    --task action_rollout \
    --epochs $EPOCHS \
    --batch_size 32 \
    --lr 1e-3 \
    --device $DEVICE \
    --use_gt_policy_as_z \
    --segment_aware \
    2>&1 | tee logs/task_a1_$RUN_TAG.log

echo "✅ Task A1 (GT Policy as z) training completed"
echo ""

# Task A2: Pretrained Policy Representation Vector as z  
echo "📋 Task A2: Pretrained Policy Representation Vector as Latent z"
echo "  - Model: Fixed-Z CVAE with learned policy representations"
echo "  - Latent: z = pretrained_policy_representation_vector"
echo "  - Segments: Known (segment-aware evaluation)"
echo ""

# First train a policy representation extractor
python baselines/state_cond/train_policy_representation_extractor.py \
    --data_root $DATA_ROOT \
    --save_dir "$REPORTS_DIR/task_a_action_output/policy_repr_extractor_$RUN_TAG" \
    --epochs $EPOCHS \
    --batch_size 32 \
    --device $DEVICE \
    2>&1 | tee logs/policy_repr_extractor_$RUN_TAG.log

# Then train CVAE using pretrained representations
python baselines/state_cond/train_cvae_fixed_z.py \
    --data_root $DATA_ROOT \
    --save_dir "$REPORTS_DIR/task_a_action_output/a2_pretrained_repr_as_z_$RUN_TAG" \
    --variant learned_repr \
    --task action_rollout \
    --epochs $EPOCHS \
    --batch_size 32 \
    --lr 1e-3 \
    --device $DEVICE \
    --pretrained_repr_path "$REPORTS_DIR/task_a_action_output/policy_repr_extractor_$RUN_TAG/model_best.pt" \
    --segment_aware \
    2>&1 | tee logs/task_a2_$RUN_TAG.log

echo "✅ Task A2 (Pretrained Repr as z) training completed"
echo ""

# =============================================================================
# TASK B: POLICY REPRESENTATION (Mixed segment awareness)
# =============================================================================
echo "🎯 TASK B: POLICY REPRESENTATION (Mixed Segment Awareness)"
echo "B1 knows segments, B2 does not know segments"
echo ""

# Task B1: Policy ID Classification (Segment-Aware)
echo "📋 Task B1: Policy ID Classification"
echo "  - Task: Classify ground truth policy ID"
echo "  - Segments: Known (segment-aware evaluation)"
echo "  - Metrics: Classification accuracy, F1-score"
echo ""

python eval/policy_classification.py \
    --data_root $DATA_ROOT \
    --model_type "fixed_z_cvae" \
    --save_dir "$REPORTS_DIR/task_b_policy_representation/b1_policy_classification_$RUN_TAG" \
    --segment_aware \
    --task policy_classification \
    --epochs $EPOCHS \
    --device $DEVICE \
    2>&1 | tee logs/task_b1_$RUN_TAG.log

echo "✅ Task B1 (Policy Classification - Segment Aware) completed"
echo ""

# Task B2: Policy Changepoint Detection (Segment-Unaware)
echo "📋 Task B2: Policy Changepoint Detection"
echo "  - Task: Detect policy changes without knowing segment boundaries"
echo "  - Segments: Unknown (segment-unaware evaluation)"
echo "  - Metrics: F1@τ, MABE, detection delay"
echo ""

python eval/policy_changepoint_detection.py \
    --data_root $DATA_ROOT \
    --model_type "fixed_z_cvae" \
    --save_dir "$REPORTS_DIR/task_b_policy_representation/b2_changepoint_detection_$RUN_TAG" \
    --segment_unaware \
    --task changepoint_detection \
    --tau 3 \
    --window_size 10 \
    --device $DEVICE \
    2>&1 | tee logs/task_b2_$RUN_TAG.log

echo "✅ Task B2 (Changepoint Detection - Segment Unaware) completed"
echo ""

# =============================================================================
# EVALUATION ACROSS ALL SHIFTS
# =============================================================================
echo "📊 Evaluating all tasks across distribution shifts"
echo ""

# Get list of all OOD splits
SPLITS=($(ls -d $DATA_ROOT/ood_* | xargs -n 1 basename))
echo "Found OOD splits: ${SPLITS[@]}"

# Evaluate Task A models on all splits
echo "🔍 Evaluating Task A: Action Output"
for split in "${SPLITS[@]}"; do
    echo "  Evaluating on $split..."
    
    # A1: GT Policy as z
    python eval/rollout.py \
        --data_root "$DATA_ROOT/$split" \
        --model "$REPORTS_DIR/task_a_action_output/a1_gt_policy_as_z_$RUN_TAG/model_best.pt" \
        --episodes 20 \
        --device $DEVICE \
        --segment_aware \
        --use_gt_policy_as_z \
        --save_json "$REPORTS_DIR/task_a_action_output/a1_results_${split}.json" \
        2>&1 | tee logs/eval_a1_${split}_$RUN_TAG.log
    
    # A2: Pretrained Repr as z  
    python eval/rollout.py \
        --data_root "$DATA_ROOT/$split" \
        --model "$REPORTS_DIR/task_a_action_output/a2_pretrained_repr_as_z_$RUN_TAG/model_best.pt" \
        --episodes 20 \
        --device $DEVICE \
        --segment_aware \
        --use_pretrained_repr_as_z \
        --repr_model "$REPORTS_DIR/task_a_action_output/policy_repr_extractor_$RUN_TAG/model_best.pt" \
        --save_json "$REPORTS_DIR/task_a_action_output/a2_results_${split}.json" \
        2>&1 | tee logs/eval_a2_${split}_$RUN_TAG.log
done

# Evaluate Task B models on all splits  
echo "🔍 Evaluating Task B: Policy Representation"
for split in "${SPLITS[@]}"; do
    echo "  Evaluating on $split..."
    
    # B1: Policy Classification (Segment-Aware)
    python eval/policy_classification.py \
        --data_root "$DATA_ROOT/$split" \
        --model "$REPORTS_DIR/task_b_policy_representation/b1_policy_classification_$RUN_TAG/model_best.pt" \
        --segment_aware \
        --save_json "$REPORTS_DIR/task_b_policy_representation/b1_results_${split}.json" \
        2>&1 | tee logs/eval_b1_${split}_$RUN_TAG.log
    
    # B2: Changepoint Detection (Segment-Unaware)
    python eval/policy_changepoint_detection.py \
        --data_root "$DATA_ROOT/$split" \
        --model "$REPORTS_DIR/task_b_policy_representation/b2_changepoint_detection_$RUN_TAG/model_best.pt" \
        --segment_unaware \
        --save_json "$REPORTS_DIR/task_b_policy_representation/b2_results_${split}.json" \
        2>&1 | tee logs/eval_b2_${split}_$RUN_TAG.log
done

echo "✅ All task evaluations completed"
echo ""

# =============================================================================
# GENERATE PROPER TASK COMPARISON PLOTS
# =============================================================================
echo "🎨 Generating proper task comparison plots"
echo ""

python scripts/create_proper_task_plots.py \
    --results_dir $REPORTS_DIR \
    --data_root $DATA_ROOT \
    --output_dir $REPORTS_DIR/plots \
    --run_tag $RUN_TAG \
    2>&1 | tee logs/plotting_proper_tasks_$RUN_TAG.log

echo "✅ Proper task comparison plots generated"
echo ""

# =============================================================================
# FINAL SUMMARY
# =============================================================================
echo "🎉 Policy-or-Proxy v5.0 Proper Task Framework Complete!"
echo ""
echo "🎯 Implemented Task Structure:"
echo ""
echo "📋 TASK A: ACTION OUTPUT (Both Segment-Aware)"
echo "  A1: Ground Truth Policy as z"
echo "    • Model: Fixed-Z CVAE with policy ID embedding"
echo "    • Segments: Known (segment-aware evaluation)"
echo "    • Metrics: ADE, FDE, collision rate, smoothness"
echo ""
echo "  A2: Pretrained Policy Representation Vector as z"
echo "    • Model: Fixed-Z CVAE with learned representations"
echo "    • Segments: Known (segment-aware evaluation)"  
echo "    • Metrics: ADE, FDE, collision rate, smoothness"
echo ""
echo "📋 TASK B: POLICY REPRESENTATION (Mixed Segment Awareness)"
echo "  B1: Policy ID Classification"
echo "    • Task: Classify ground truth policy ID"
echo "    • Segments: Known (segment-aware evaluation)"
echo "    • Metrics: Classification accuracy, F1-score"
echo ""
echo "  B2: Policy Changepoint Detection"
echo "    • Task: Detect policy changes without segment info"
echo "    • Segments: Unknown (segment-unaware evaluation)"
echo "    • Metrics: F1@τ, MABE, detection delay"
echo ""
echo "📁 Generated Outputs:"
echo "  • Task A Results: $REPORTS_DIR/task_a_action_output/"
echo "  • Task B Results: $REPORTS_DIR/task_b_policy_representation/"
echo "  • Comparison Plots: $REPORTS_DIR/plots/"
echo "  • Evaluation Logs: logs/"
echo ""
echo "🔍 Key Comparisons:"
echo "  1. A1 vs A2: GT Policy z vs Pretrained Repr z (action output)"
echo "  2. B1 vs B2: Known segments vs Unknown segments (policy representation)"
echo "  3. All tasks across distribution shifts (robustness analysis)"
echo ""
echo "Run completed at: $(date)"
