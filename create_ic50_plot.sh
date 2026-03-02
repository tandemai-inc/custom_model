#!/bin/bash
# Script to create CV predictions plot for peptide_IC50 after training completes

cd "/Users/chrishe/Documents/Untitled Folder 3/xgboost_training"

# Wait for training to complete
while [ ! -f "peptide_IC50_results/training_results.json" ]; do
    echo "Waiting for training to complete..."
    sleep 60
done

echo "Training completed! Creating plot..."

source $(conda info --base)/etc/profile.d/conda.sh
conda activate xgboost_training

python plot_cv_predictions_custom.py \
    --cv_predictions peptide_IC50_results/cv_all_folds_predictions.csv \
    --output peptide_IC50_results/cv_predictions_plot.png \
    --title "Peptide IC50 - Cross-Validation Predictions"

echo "Plot created at peptide_IC50_results/cv_predictions_plot.png"
