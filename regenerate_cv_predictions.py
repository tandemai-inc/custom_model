"""
Re-run cross-validation on an existing model to generate prediction files.

This is useful when training was done before the prediction saving feature was added.
"""

import pandas as pd
import numpy as np
import pickle
import json
import os
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from xgboost_trainer import cross_validate_model


def main():
    parser = argparse.ArgumentParser(
        description='Re-run CV on existing model to generate prediction files'
    )
    parser.add_argument(
        '--model_path',
        type=str,
        required=True,
        help='Path to saved model (.pkl file)'
    )
    parser.add_argument(
        '--data',
        type=str,
        required=True,
        help='Path to data CSV file'
    )
    parser.add_argument(
        '--hyperparameters',
        type=str,
        required=True,
        help='Path to best_hyperparameters.json file'
    )
    parser.add_argument(
        '--selected_features',
        type=str,
        default=None,
        help='Path to selected_features.json (if feature selection was used)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='Output directory to save prediction files'
    )
    parser.add_argument(
        '--task',
        type=str,
        default='regression',
        choices=['regression', 'classification'],
        help='Task type'
    )
    parser.add_argument(
        '--cv_folds',
        type=int,
        default=5,
        help='Number of CV folds'
    )
    parser.add_argument(
        '--random_state',
        type=int,
        default=42,
        help='Random state'
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("RE-GENERATING CV PREDICTIONS")
    print("=" * 80)
    print(f"Model: {args.model_path}")
    print(f"Data: {args.data}")
    print(f"Output: {args.output_dir}")
    print("=" * 80)
    
    # Load data
    print("\n1. Loading data...")
    df = pd.read_csv(args.data)
    print(f"   Loaded {len(df)} samples")
    
    if 'smiles' not in df.columns or 'label' not in df.columns:
        print("Error: Data must contain 'smiles' and 'label' columns")
        return
    
    # Extract features and labels
    if args.selected_features:
        print(f"\n2. Loading selected features from {args.selected_features}...")
        with open(args.selected_features, 'r') as f:
            selected_data = json.load(f)
            selected_feature_names = selected_data['selected_features']
        
        # Get feature columns (exclude metadata)
        metadata_cols = ['smiles', 'label', 'fold'] if 'fold' in df.columns else ['smiles', 'label']
        feature_cols = [col for col in df.columns if col not in metadata_cols]
        
        # Filter to selected features
        available_features = [f for f in selected_feature_names if f in feature_cols]
        print(f"   Using {len(available_features)} selected features")
        X = df[available_features]
    else:
        # Use all feature columns
        metadata_cols = ['smiles', 'label', 'fold'] if 'fold' in df.columns else ['smiles', 'label']
        feature_cols = [col for col in df.columns if col not in metadata_cols]
        X = df[feature_cols]
        print(f"\n2. Using {len(feature_cols)} features")
    
    y = df['label']
    
    # Handle classification threshold if needed
    if args.task == 'classification':
        # Check if labels are already binary
        if y.dtype in ['int', 'int64'] and y.nunique() == 2:
            print("   Labels are already binary")
        else:
            threshold = y.median()
            print(f"   Applying median split threshold: {threshold:.4f}")
            y = (y >= threshold).astype(int)
    
    # Load hyperparameters
    print(f"\n3. Loading hyperparameters from {args.hyperparameters}...")
    with open(args.hyperparameters, 'r') as f:
        params = json.load(f)
    print(f"   Loaded hyperparameters")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Run cross-validation
    print(f"\n4. Running cross-validation...")
    print(f"   Folds: {args.cv_folds}")
    print(f"   Task: {args.task}")
    
    cv_results = cross_validate_model(
        X, y,
        params,
        task=args.task,
        cv_folds=args.cv_folds,
        random_state=args.random_state,
        output_dir=args.output_dir
    )
    
    print("\n" + "=" * 80)
    print("CV PREDICTIONS GENERATED")
    print("=" * 80)
    print(f"Prediction files saved to: {args.output_dir}")
    print(f"  - Individual folds: cv_fold_1_predictions.csv, ...")
    print(f"  - Combined: cv_all_folds_predictions.csv")
    print("=" * 80)


if __name__ == "__main__":
    main()

