# XGBoost Training Module with Optuna Hyperparameter Optimization

This module provides a complete pipeline for training XGBoost models on molecular data using multiple feature types and Optuna for hyperparameter optimization. Supports both regression and classification tasks with feature selection capabilities.

## Features

- **Multiple Feature Types**: Combines reduced Mordred descriptors, Morgan fingerprints, RDKit fingerprints, MACCS keys, and MAP4 fingerprints
- **Peptide Data Support**: Concatenate peptide embeddings with molecular features for combined training
- **Feature Selection**: Mutual information-based feature selection to prevent overfitting
- **Task Support**: Both regression and classification (with median split thresholding)
- **Optuna Optimization**: Uses cross-validation for robust hyperparameter selection
- **Comprehensive Evaluation**: Includes test set evaluation and cross-validation with multiple metrics
- **Feature Importance**: Extracts and ranks feature importance
- **Model Persistence**: Saves trained models and results

## Installation

### Option 1: Conda Environment (Recommended)

```bash
# Create and activate conda environment
conda env create -f environment.yml
conda activate xgboost_training

# Or use the setup script (Linux/Mac)
./setup_conda.sh

# Or use the setup script (Windows)
setup_conda.bat
```

### Option 2: Pip Installation

```bash
pip install -r requirements.txt
```

**Note**: Some packages (like `rdkit-pypi` and `mordred`) may be easier to install via conda.

## Usage

### Basic Regression Training

```bash
python train_model.py \
    --data train_val_features.csv \
    --task regression \
    --n_trials 100 \
    --cv_folds 5 \
    --metric r2 \
    --output_dir results/
```

### Classification Training

```bash
python train_model.py \
    --data ../herg_class.csv \
    --task classification \
    --use_feature_selection \
    --feature_selection_ratio 1.0 \
    --n_trials 50 \
    --output_dir classification_results/
```

### Training with Feature Selection

```bash
python train_model.py \
    --data train_val_features.csv \
    --task regression \
    --use_feature_selection \
    --feature_selection_ratio 1.0 \
    --n_trials 50 \
    --output_dir results_with_selection/
```

### Training on All Data (No Test Set)

```bash
python train_model.py \
    --data peptide_molecular_combined.csv \
    --task regression \
    --use_feature_selection \
    --feature_selection_ratio 1.0 \
    --test_size 0 \
    --n_trials 50 \
    --output_dir results_all_data/
```

### Concatenating Peptide Embeddings with Molecular Features

```bash
# Concatenate peptide embeddings with molecular features
python concatenate_peptide_features.py \
    --peptide_data ~/Downloads/processed_data_filtered_with_embeddings.csv \
    --output peptide_molecular_combined.csv \
    --reduced_features reduced_mordred_features.json

# Then train on the combined dataset
python train_model.py \
    --data peptide_molecular_combined.csv \
    --task regression \
    --use_feature_selection \
    --feature_selection_ratio 1.0 \
    --n_trials 50 \
    --output_dir peptide_results/
```

## Workflow Orchestration (ta_workflow)

For job scheduling and monitoring via ta_workflow (similar to ta_admet):

```bash
# Training
python workflows/xgboost_workflow_master.py -i /path/to/config.yaml

# Config must define parameter.xgboost_training or parameter.xgboost_prediction
# See workflows/config/example_training_config.yaml and example_prediction_config.yaml
```

**Dependencies**: `ta_workflow`, `ta_base`, `pyyaml`. Install from your internal package index.

## Command Line Arguments

### `train_model.py`

- `--data`: Path to training data CSV file (default: `train_val_features.csv`)
- `--task`: Task type - `regression` or `classification` (default: `regression`)
- `--classification_threshold`: Custom threshold for binary classification (optional, defaults to median)
- `--n_trials`: Number of Optuna trials (default: 100)
- `--cv_folds`: Number of CV folds (default: 5)
- `--metric`: Metric to optimize - `r2`, `mae`, `rmse` for regression; `accuracy`, `f1`, `roc_auc` for classification (default: `r2`)
- `--output_dir`: Output directory for results (default: `results`)
- `--reduced_features`: Path to reduced Mordred features JSON (default: `reduced_mordred_features.json`)
- `--include_map4`: Include MAP4 fingerprints (default: True)
- `--map4_dimensions`: MAP4 fingerprint dimensions (default: 1024)
- `--test_size`: Test set size (0.0 to 1.0, or 0 for all data) (default: 0.2)
- `--random_state`: Random state for reproducibility (default: 42)
- `--skip_optuna`: Skip Optuna optimization and use default hyperparameters
- `--use_feature_selection`: Enable mutual information-based feature selection
- `--feature_selection_ratio`: Ratio of features to samples (e.g., 1.0 means n_features = n_samples)
- `--feature_selection_n`: Fixed number of features to keep (overrides ratio)
- `--feature_selection_threshold`: Mutual information score threshold
- `--saved_features`: Path to pre-selected features JSON file

### `concatenate_peptide_features.py`

- `--peptide_data`: Path to peptide data CSV with embeddings (default: `~/Downloads/processed_data_filtered_with_embeddings.csv`)
- `--output`: Output CSV file path (default: `peptide_molecular_combined.csv`)
- `--reduced_features`: Path to reduced Mordred features JSON (default: `reduced_mordred_features.json`)
- `--include_map4`: Include MAP4 fingerprints (default: True)
- `--map4_dimensions`: MAP4 fingerprint dimensions (default: 1024)
- `--save_features_only`: Save only features (no labels/metadata)

## Module Structure

### `featurization.py`
Calculates and combines all molecular features:
- Reduced Mordred descriptors
- Morgan fingerprints (radius=2, 2048 bits)
- RDKit fingerprints (Atom Pair, 2048 bits)
- MACCS keys (167 bits)
- MAP4 fingerprints (configurable dimensions)

### `feature_selection.py`
Mutual information-based feature selection:
- Supports both regression and classification
- Multiple selection criteria (ratio, fixed number, threshold)
- Saves selected features for reuse

### `optuna_tuning.py`
Optuna hyperparameter optimization with cross-validation:
- Defines hyperparameter search space
- Uses CV scores for robust optimization
- Supports regression and classification metrics
- Task-specific model instantiation

### `xgboost_trainer.py`
XGBoost model training and evaluation:
- Model training with early stopping (XGBoost 2.0+ compatible)
- Model evaluation with comprehensive metrics:
  - Regression: R², MAE, RMSE, Pearson r, Spearman ρ
  - Classification: Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC, Confusion Matrix
- Cross-validation
- Model persistence (save/load)
- Feature importance extraction (handles both feature names and indices)

### `train_model.py`
Main orchestration script that:
1. Loads data
2. Detects/sets task type (regression/classification)
3. Applies classification threshold if needed
4. Calculates features
5. Performs feature selection (if enabled)
6. Splits data (or uses all data if test_size=0)
7. Runs Optuna optimization
8. Trains final model
9. Evaluates on test set (if available)
10. Runs cross-validation
11. Extracts feature importance
12. Saves model and results

### `concatenate_peptide_features.py`
Concatenates peptide embeddings with molecular features:
1. Loads peptide data with pre-computed embeddings
2. Calculates molecular features from SMILES
3. Concatenates embeddings + molecular features
4. Saves combined dataset for training

## Output Files

The training pipeline generates the following files in the output directory:

- `best_model.pkl`: Trained XGBoost model
- `best_hyperparameters.json`: Optimal hyperparameters from Optuna
- `training_results.json`: Complete training results and metrics
- `feature_importance.csv`: Feature importance rankings
- `selected_features.json`: Selected features (if feature selection used)
- `feature_selection_scores.csv`: Mutual information scores for all features
- `confusion_matrix.csv`: Confusion matrix (classification only)
- `classification_report.txt`: Classification report (classification only)

## Examples

### Regression with Feature Selection

```bash
python train_model.py \
    --data ../train_val_features_1000.csv \
    --task regression \
    --use_feature_selection \
    --feature_selection_ratio 1.0 \
    --n_trials 50 \
    --output_dir regression_with_selection/
```

### Classification with Median Split

```bash
python train_model.py \
    --data ../herg_class.csv \
    --task classification \
    --use_feature_selection \
    --feature_selection_ratio 1.0 \
    --n_trials 50 \
    --output_dir classification_results/
```

### Peptide + Molecular Features

```bash
# Step 1: Concatenate features
python concatenate_peptide_features.py \
    --peptide_data ~/Downloads/processed_data_filtered_with_embeddings.csv \
    --output peptide_molecular_combined.csv

# Step 2: Train model
python train_model.py \
    --data peptide_molecular_combined.csv \
    --task regression \
    --use_feature_selection \
    --feature_selection_ratio 1.0 \
    --n_trials 50 \
    --output_dir peptide_results/
```

### Training on All Data

```bash
python train_model.py \
    --data peptide_molecular_combined.csv \
    --task regression \
    --use_feature_selection \
    --feature_selection_ratio 1.0 \
    --test_size 0 \
    --n_trials 50 \
    --output_dir results_all_data/
```

## Monitoring Training Progress

Use the monitoring script to check training progress:

```bash
python monitor_training.py
```

Or use the shell script:

```bash
./check_training_progress.sh
```

## Dependencies

- pandas
- numpy
- scikit-learn
- xgboost
- optuna
- rdkit-pypi
- mordred
- map4
- scipy (for correlation metrics)

## Notes

- Ensure `reduced_mordred_features.json` exists in the current directory (or specify path with `--reduced_features`)
- MAP4 requires the `map4` package (install with `pip install map4`)
- The pipeline handles missing values by filling with 0
- Early stopping is used during training to prevent overfitting
- Feature selection is recommended when features > samples to prevent overfitting
- Classification uses median split by default, but custom thresholds can be specified
- When `test_size=0`, all data is used for training and only CV metrics are available
