#!/bin/bash

# Simplified Demo: Proper Task Framework
# Demonstrates the correct task structure with minimal components

set -e

RUN_TAG="demo_proper_$(date +%Y%m%d_%H%M%S)"
DATA_ROOT="data/v5.0_simple"  # Use existing v5.0 dataset
REPORTS_DIR="reports/v5.0_proper_demo"
DEVICE="cpu"
EPOCHS=3  # Very small for demo

echo "🎯 Demonstrating Proper Task Framework"
echo "Task A: Action Output (A1: GT Policy z, A2: Pretrained Repr z)"
echo "Task B: Policy Representation (B1: Classification, B2: Changepoint)"
echo ""

# Create directories
mkdir -p $REPORTS_DIR/task_a_action_output
mkdir -p $REPORTS_DIR/task_b_policy_representation
mkdir -p logs

# Verify dataset exists
if [ ! -d "$DATA_ROOT" ]; then
    echo "❌ Dataset not found at $DATA_ROOT. Please run v5.0 data generation first."
    exit 1
fi

echo "📊 Using existing v5.0 dataset: $DATA_ROOT"
echo ""

# =============================================================================
# DEMO: TASK A - ACTION OUTPUT 
# =============================================================================
echo "🎯 TASK A DEMO: ACTION OUTPUT (Both Segment-Aware)"
echo ""

# A1: Ground Truth Policy as z (simplified demonstration)
echo "📋 Task A1 Demo: GT Policy ID as Latent z"
echo "  - Training Fixed-Z CVAE with policy ID conditioning"
echo "  - Segment-aware: Model knows fixed policy segments"
echo ""

python baselines/state_cond/train_cvae_fixed_z.py \
    --data_root $DATA_ROOT \
    --save_dir "$REPORTS_DIR/task_a_action_output/a1_demo_$RUN_TAG" \
    --variant policy_conditional \
    --task action_rollout \
    --epochs $EPOCHS \
    --batch_size 32 \
    --device $DEVICE \
    --use_gt_policy_as_z \
    --segment_aware \
    2>&1 | tee logs/demo_task_a1_$RUN_TAG.log

echo "✅ Task A1 demo completed"
echo ""

# A2: Pretrained Policy Representation (simplified - will use same model for demo)
echo "📋 Task A2 Demo: Pretrained Representation as z"
echo "  - Training Fixed-Z CVAE with learned representations"
echo "  - Segment-aware: Model knows fixed policy segments"
echo ""

python baselines/state_cond/train_cvae_fixed_z.py \
    --data_root $DATA_ROOT \
    --save_dir "$REPORTS_DIR/task_a_action_output/a2_demo_$RUN_TAG" \
    --variant learned_repr \
    --task action_rollout \
    --epochs $EPOCHS \
    --batch_size 32 \
    --device $DEVICE \
    --segment_aware \
    2>&1 | tee logs/demo_task_a2_$RUN_TAG.log

echo "✅ Task A2 demo completed"
echo ""

# =============================================================================
# DEMO: TASK B - POLICY REPRESENTATION
# =============================================================================
echo "🎯 TASK B DEMO: POLICY REPRESENTATION (Mixed Segment Awareness)"
echo ""

# B1: Policy Classification (Segment-Aware)
echo "📋 Task B1 Demo: Policy Classification (Segment-Aware)"
echo "  - Training policy classifier with known segments"
echo "  - Segment-aware: Model knows the policy segment boundaries"
echo ""

python eval/policy_classification.py \
    --data_root $DATA_ROOT \
    --save_dir "$REPORTS_DIR/task_b_policy_representation/b1_demo_$RUN_TAG" \
    --segment_aware \
    --epochs $EPOCHS \
    --device $DEVICE \
    2>&1 | tee logs/demo_task_b1_$RUN_TAG.log

echo "✅ Task B1 demo completed"
echo ""

# B2: Changepoint Detection (Segment-Unaware)
echo "📋 Task B2 Demo: Changepoint Detection (Segment-Unaware)"
echo "  - Training changepoint detector without segment info"
echo "  - Segment-unaware: Model must discover policy changes"
echo ""

python eval/policy_changepoint_detection.py \
    --data_root $DATA_ROOT \
    --save_dir "$REPORTS_DIR/task_b_policy_representation/b2_demo_$RUN_TAG" \
    --segment_unaware \
    --epochs $EPOCHS \
    --device $DEVICE \
    2>&1 | tee logs/demo_task_b2_$RUN_TAG.log

echo "✅ Task B2 demo completed"
echo ""

# =============================================================================
# DEMO SUMMARY
# =============================================================================
echo "🎉 Proper Task Framework Demo Complete!"
echo ""
echo "🎯 Demonstrated Task Structure:"
echo ""
echo "📋 TASK A: ACTION OUTPUT (Both Segment-Aware)"
echo "  ✅ A1: Ground Truth Policy as z"
echo "    • Model: Fixed-Z CVAE with policy ID embedding"
echo "    • Latent: z = policy_embedding[gt_policy_id]"
echo "    • Segments: Known (segment-aware evaluation)"
echo ""
echo "  ✅ A2: Pretrained Policy Representation as z" 
echo "    • Model: Fixed-Z CVAE with learned representations"
echo "    • Latent: z = pretrained_policy_repr"
echo "    • Segments: Known (segment-aware evaluation)"
echo ""
echo "📋 TASK B: POLICY REPRESENTATION (Mixed Segment Awareness)"
echo "  ✅ B1: Policy ID Classification"
echo "    • Task: Classify ground truth policy ID"
echo "    • Segments: Known (segment-aware evaluation)"
echo "    • Metrics: Classification accuracy, F1-score"
echo ""
echo "  ✅ B2: Policy Changepoint Detection"
echo "    • Task: Detect policy changes without segment info"
echo "    • Segments: Unknown (segment-unaware evaluation)"
echo "    • Metrics: F1@τ, MABE, detection delay"
echo ""
echo "🔍 Key Comparisons Enabled:"
echo "  1. A1 vs A2: GT Policy z vs Pretrained Repr z (action output)"
echo "  2. B1 vs B2: Known segments vs Unknown segments (policy representation)"
echo "  3. Task A vs Task B: Action output vs Policy representation"
echo "  4. Segment-aware vs Segment-unaware: Impact of segment knowledge"
echo ""
echo "📁 Demo Results: $REPORTS_DIR"
echo "📜 Demo Logs: logs/"
echo ""
echo "✨ Framework ready for full experimental comparison!"
