#!/bin/bash
# Quick script to check training progress
# Can be run manually or via cron

cd "$(dirname "$0")"
OUTPUT_DIR="${1:-permeability_with_embeddings}"

echo "================================================================================"
echo "TRAINING PROGRESS CHECK: $OUTPUT_DIR"
echo "================================================================================"
echo "Time: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

# Check if process is running
if ps aux | grep -v grep | grep -q "train_model.py.*permeability"; then
    echo "✓ Training process is running"
    ps aux | grep -v grep | grep "train_model.py.*permeability" | head -1 | awk '{print "  CPU: "$3"% | Memory: "$6/1024"MB"}'
else
    echo "⚠️  Training process not found"
fi

echo ""

# Check results directory
if [ -d "$OUTPUT_DIR" ]; then
    echo "Results directory: $OUTPUT_DIR"
    echo "Files:"
    ls -lht "$OUTPUT_DIR" 2>/dev/null | head -10 | awk '{print "  " $9 " (" $5 ")"}'
    
    echo ""
    
    # Check for completion
    if [ -f "$OUTPUT_DIR/training_results.json" ]; then
        echo "🎉 TRAINING COMPLETE! 🎉"
        echo ""
        conda run -n xgboost_training python3 << EOF
import json
with open('$OUTPUT_DIR/training_results.json', 'r') as f:
    results = json.load(f)
if 'cv_metrics' in results:
    cv = results['cv_metrics']
    if 'r2_mean' in cv:
        print(f"CV R²: {cv['r2_mean']:.4f} ± {cv['r2_std']:.4f}")
        print(f"CV RMSE: {cv['rmse_mean']:.4f} ± {cv['rmse_std']:.4f}")
EOF
    elif [ -f "$OUTPUT_DIR/best_hyperparameters.json" ]; then
        echo "⏳ Optuna complete, final training in progress..."
    elif [ -f "$OUTPUT_DIR/selected_features.json" ]; then
        echo "⏳ Feature selection complete, Optuna optimization running..."
        echo "   Estimated time remaining: 1-3 hours"
    else
        echo "⏳ Early stages (feature selection or initialization)..."
    fi
else
    echo "⏳ Results directory not created yet - training initializing..."
fi

echo ""
echo "================================================================================"
