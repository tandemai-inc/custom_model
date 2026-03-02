"""
Comprehensive model evaluation script.

Calculates multiple metrics including:
- Spearman correlation
- Pearson correlation
- R², MAE, RMSE
- Median Absolute Error
- Mean Absolute Percentage Error (MAPE)
- Explained Variance Score
"""

import pandas as pd
import numpy as np
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import (
    r2_score, mean_absolute_error, mean_squared_error,
    median_absolute_error, explained_variance_score
)
import json
import argparse
import os
import sys
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.xgboost_trainer import load_model, evaluate_model


def calculate_all_metrics(y_true, y_pred):
    """
    Calculate comprehensive evaluation metrics.
    
    Args:
        y_true: True target values
        y_pred: Predicted target values
        
    Returns:
        Dictionary with all metrics
    """
    # Remove any NaN or inf values
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true_clean = y_true[mask]
    y_pred_clean = y_pred[mask]
    
    if len(y_true_clean) == 0:
        return {'error': 'No valid predictions'}
    
    metrics = {}
    
    # Basic regression metrics
    metrics['r2'] = r2_score(y_true_clean, y_pred_clean)
    metrics['mae'] = mean_absolute_error(y_true_clean, y_pred_clean)
    metrics['rmse'] = np.sqrt(mean_squared_error(y_true_clean, y_pred_clean))
    metrics['mse'] = mean_squared_error(y_true_clean, y_pred_clean)
    metrics['median_ae'] = median_absolute_error(y_true_clean, y_pred_clean)
    metrics['explained_variance'] = explained_variance_score(y_true_clean, y_pred_clean)
    
    # Correlation metrics
    pearson_corr, pearson_p = pearsonr(y_true_clean, y_pred_clean)
    metrics['pearson_r'] = pearson_corr
    metrics['pearson_p_value'] = pearson_p
    
    spearman_corr, spearman_p = spearmanr(y_true_clean, y_pred_clean)
    metrics['spearman_r'] = spearman_corr
    metrics['spearman_p_value'] = spearman_p
    
    # Additional metrics
    metrics['mean_absolute_percentage_error'] = np.mean(
        np.abs((y_true_clean - y_pred_clean) / (y_true_clean + 1e-10))
    ) * 100
    
    # Residual statistics
    residuals = y_true_clean - y_pred_clean
    metrics['mean_residual'] = np.mean(residuals)
    metrics['std_residual'] = np.std(residuals)
    metrics['median_residual'] = np.median(residuals)
    
    # Target statistics for context
    metrics['target_mean'] = np.mean(y_true_clean)
    metrics['target_std'] = np.std(y_true_clean)
    metrics['target_range'] = [float(np.min(y_true_clean)), float(np.max(y_true_clean))]
    
    # Prediction statistics
    metrics['pred_mean'] = np.mean(y_pred_clean)
    metrics['pred_std'] = np.std(y_pred_clean)
    metrics['pred_range'] = [float(np.min(y_pred_clean)), float(np.max(y_pred_clean))]
    
    # Relative errors
    metrics['mae_relative'] = metrics['mae'] / metrics['target_mean'] * 100 if metrics['target_mean'] != 0 else np.nan
    metrics['rmse_relative'] = metrics['rmse'] / metrics['target_mean'] * 100 if metrics['target_mean'] != 0 else np.nan
    
    return metrics


def evaluate_model_comprehensive(model, X, y, dataset_name="Dataset"):
    """
    Comprehensive model evaluation.
    
    Args:
        model: Trained XGBoost model
        X: Features
        y: True targets
        dataset_name: Name for display
        
    Returns:
        Dictionary with all metrics
    """
    print(f"\n{'='*80}")
    print(f"Comprehensive Evaluation: {dataset_name}")
    print(f"{'='*80}")
    
    # Handle NaN values
    X = X.fillna(0)
    
    # Make predictions
    y_pred = model.predict(X)
    
    # Calculate all metrics
    metrics = calculate_all_metrics(y.values, y_pred)
    
    # Display results
    print(f"\n📊 Regression Metrics:")
    print(f"  R² Score:                    {metrics['r2']:.6f}")
    print(f"  Explained Variance:          {metrics['explained_variance']:.6f}")
    print(f"  Mean Absolute Error (MAE):   {metrics['mae']:.6f}")
    print(f"  Median Absolute Error:       {metrics['median_ae']:.6f}")
    print(f"  Root Mean Squared Error:     {metrics['rmse']:.6f}")
    print(f"  Mean Squared Error (MSE):    {metrics['mse']:.6f}")
    print(f"  Mean Absolute % Error:       {metrics['mean_absolute_percentage_error']:.2f}%")
    
    print(f"\n📈 Correlation Metrics:")
    print(f"  Pearson Correlation (r):     {metrics['pearson_r']:.6f} (p={metrics['pearson_p_value']:.2e})")
    print(f"  Spearman Correlation (ρ):     {metrics['spearman_r']:.6f} (p={metrics['spearman_p_value']:.2e})")
    
    print(f"\n📉 Residual Statistics:")
    print(f"  Mean Residual:                {metrics['mean_residual']:.6f}")
    print(f"  Std Residual:                 {metrics['std_residual']:.6f}")
    print(f"  Median Residual:              {metrics['median_residual']:.6f}")
    
    print(f"\n📊 Target Statistics:")
    print(f"  Mean:                         {metrics['target_mean']:.6f}")
    print(f"  Std:                          {metrics['target_std']:.6f}")
    print(f"  Range:                        [{metrics['target_range'][0]:.2f}, {metrics['target_range'][1]:.2f}]")
    
    print(f"\n📊 Prediction Statistics:")
    print(f"  Mean:                         {metrics['pred_mean']:.6f}")
    print(f"  Std:                          {metrics['pred_std']:.6f}")
    print(f"  Range:                        [{metrics['pred_range'][0]:.2f}, {metrics['pred_range'][1]:.2f}]")
    
    print(f"\n📊 Relative Errors:")
    if not np.isnan(metrics['mae_relative']):
        print(f"  MAE (relative to mean):       {metrics['mae_relative']:.2f}%")
    if not np.isnan(metrics['rmse_relative']):
        print(f"  RMSE (relative to mean):      {metrics['rmse_relative']:.2f}%")
    
    return metrics


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description='Comprehensive model evaluation')
    parser.add_argument('--model', type=str, default='results/best_model.pkl',
                       help='Path to trained model (default: results/best_model.pkl)')
    parser.add_argument('--data', type=str, required=True,
                       help='Path to test data CSV file')
    parser.add_argument('--output', type=str, default='evaluation_results.json',
                       help='Output file for metrics (default: evaluation_results.json)')
    parser.add_argument('--reduced_features', type=str,
                       default='reduced_mordred_features.json',
                       help='Path to reduced Mordred features JSON')
    parser.add_argument('--include_map4', action='store_true', default=True,
                       help='Include MAP4 fingerprints')
    parser.add_argument('--map4_dimensions', type=int, default=1024,
                       help='MAP4 fingerprint dimensions')
    
    args = parser.parse_args()
    
    print("="*80)
    print("Comprehensive Model Evaluation")
    print("="*80)
    
    # Load model
    print(f"\n1. Loading model from {args.model}...")
    try:
        model = load_model(args.model)
    except FileNotFoundError:
        print(f"Error: Model file not found: {args.model}")
        return
    
    # Load data
    print(f"\n2. Loading data from {args.data}...")
    try:
        df = pd.read_csv(args.data)
        print(f"   Loaded {len(df)} samples")
    except FileNotFoundError:
        print(f"Error: Data file not found: {args.data}")
        return
    
    # Extract SMILES and labels
    if 'smiles' not in df.columns or 'label' not in df.columns:
        print("Error: Data must contain 'smiles' and 'label' columns")
        return
    
    smiles_list = df['smiles'].dropna().tolist()
    y = df['label'].dropna()
    
    # Align indices
    valid_indices = df.dropna(subset=['smiles', 'label']).index
    smiles_list = [df.loc[i, 'smiles'] for i in valid_indices]
    y = df.loc[valid_indices, 'label']
    
    print(f"   Valid samples: {len(smiles_list)}")
    
    # Calculate features
    print(f"\n3. Calculating features...")
    from featurization import calculate_all_features
    
    try:
        X = calculate_all_features(
            smiles_list,
            reduced_features_path=args.reduced_features,
            include_map4=args.include_map4,
            map4_dimensions=args.map4_dimensions
        )
        print(f"   Features calculated: {X.shape[1]}")
    except Exception as e:
        print(f"Error calculating features: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Evaluate
    print(f"\n4. Evaluating model...")
    metrics = evaluate_model_comprehensive(model, X, y, "Test Set")
    
    # Save results
    print(f"\n5. Saving results...")
    results = {
        'model_path': args.model,
        'data_path': args.data,
        'n_samples': len(smiles_list),
        'n_features': X.shape[1],
        'metrics': metrics
    }
    
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"   Results saved to {args.output}")
    
    # Summary
    print(f"\n{'='*80}")
    print("Evaluation Summary")
    print(f"{'='*80}")
    print(f"R² Score:        {metrics['r2']:.6f}")
    print(f"Pearson r:        {metrics['pearson_r']:.6f}")
    print(f"Spearman ρ:       {metrics['spearman_r']:.6f}")
    print(f"MAE:              {metrics['mae']:.6f}")
    print(f"RMSE:             {metrics['rmse']:.6f}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()



