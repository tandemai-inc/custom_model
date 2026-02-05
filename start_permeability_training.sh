#!/bin/bash
# Start permeability training with embeddings
# This script waits for concatenation to complete, then starts training

cd "$(dirname "$0")"

COMBINED_FILE="permeability_combined.csv"
OUTPUT_DIR="permeability_with_embeddings"

echo "================================================================================"
echo "PERMEABILITY TRAINING WITH EMBEDDINGS"
echo "================================================================================"
echo "Time: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

# Check if concatenation is complete
if [ ! -f "$COMBINED_FILE" ]; then
    echo "⏳ Waiting for concatenation to complete..."
    echo "   Checking for: $COMBINED_FILE"
    
    # Wait up to 1 hour for concatenation
    MAX_WAIT=3600
    WAITED=0
    while [ ! -f "$COMBINED_FILE" ] && [ $WAITED -lt $MAX_WAIT ]; do
        sleep 30
        WAITED=$((WAITED + 30))
        if [ $((WAITED % 300)) -eq 0 ]; then
            echo "   Still waiting... ($(($WAITED / 60)) minutes elapsed)"
        fi
    done
    
    if [ ! -f "$COMBINED_FILE" ]; then
        echo "❌ Error: Concatenation file not found after waiting"
        echo "   Please check if concatenation process is still running"
        exit 1
    fi
fi

echo "✓ Concatenation file found: $COMBINED_FILE"
ls -lh "$COMBINED_FILE"
echo ""

# Check if training is already running
if ps aux | grep -v grep | grep -q "train_model.py.*permeability"; then
    echo "⚠️  Training process already running!"
    ps aux | grep -v grep | grep "train_model.py.*permeability" | head -1
    echo ""
    read -p "Do you want to start a new training anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Exiting..."
        exit 0
    fi
fi

echo "Starting training..."
echo "  Data: $COMBINED_FILE"
echo "  Output: $OUTPUT_DIR"
echo "  Task: regression"
echo "  Feature selection: enabled (ratio=1.0)"
echo "  Optuna trials: 50"
echo "  CV folds: 5"
echo ""

# Start training in background
nohup conda run -n xgboost_training python train_model.py \
    --data "$COMBINED_FILE" \
    --task regression \
    --use_feature_selection \
    --feature_selection_ratio 1.0 \
    --n_trials 50 \
    --cv_folds 5 \
    --output_dir "$OUTPUT_DIR" \
    --random_state 42 \
    --test_size 0.2 \
    > "permeability_training.log" 2>&1 &

TRAIN_PID=$!
echo "✓ Training started (PID: $TRAIN_PID)"
echo ""
echo "Monitor progress with:"
echo "  ./check_training_progress.sh $OUTPUT_DIR"
echo "  conda run -n xgboost_training python monitor_permeability.py"
echo ""
echo "View log:"
echo "  tail -f permeability_training.log"
echo ""
echo "================================================================================"

