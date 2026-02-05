#!/bin/bash
# Script to view Optuna hyperparameter tuning logs

LOG_FILE="${1:-permeability_training_fixed.log}"
OUTPUT_DIR="${2:-permeability_with_embeddings}"

echo "================================================================================"
echo "OPTUNA HYPERPARAMETER TUNING LOGS"
echo "================================================================================"
echo ""

if [ -f "$LOG_FILE" ] && [ -s "$LOG_FILE" ]; then
    echo "1. Optuna Progress (from log file):"
    echo "   ------------------------------------"
    grep -E "(Optuna|Trial|Best|hyperparameter|optimization)" "$LOG_FILE" | tail -50
    echo ""
    
    echo "2. Trial Results:"
    echo "   ------------------------------------"
    grep -E "Trial [0-9]+ finished" "$LOG_FILE" | tail -20
    echo ""
    
    echo "3. Best Hyperparameters Found:"
    echo "   ------------------------------------"
    grep -A 20 "Best hyperparameters:" "$LOG_FILE" | head -25
    echo ""
else
    echo "⚠️  Log file not found or empty: $LOG_FILE"
    echo "   Training may still be in progress or log hasn't been written yet"
    echo ""
fi

if [ -f "$OUTPUT_DIR/best_hyperparameters.json" ]; then
    echo "4. Saved Best Hyperparameters:"
    echo "   ------------------------------------"
    cat "$OUTPUT_DIR/best_hyperparameters.json"
    echo ""
else
    echo "4. Best hyperparameters file not found yet (optimization may still be running)"
    echo ""
fi

echo "================================================================================"
echo ""
echo "To view live updates, run:"
echo "  tail -f $LOG_FILE | grep -E '(Trial|Optuna|Best)'"
echo ""
echo "To view all Optuna output:"
echo "  grep -E '(Optuna|Trial)' $LOG_FILE"

