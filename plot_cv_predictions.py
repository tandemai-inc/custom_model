"""
Script to create plots for cross-validation predictions.

Creates:
- Scatter plots (predicted vs true) for each fold and combined
- Residual plots
- Distribution plots
- Performance metrics per fold
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse
from pathlib import Path
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


def calculate_metrics(y_true, y_pred):
    """Calculate regression metrics."""
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true_clean = y_true[mask]
    y_pred_clean = y_pred[mask]
    
    if len(y_true_clean) == 0:
        return None
    
    metrics = {
        'r2': r2_score(y_true_clean, y_pred_clean),
        'mae': mean_absolute_error(y_true_clean, y_pred_clean),
        'rmse': np.sqrt(mean_squared_error(y_true_clean, y_pred_clean)),
        'pearson_r': pearsonr(y_true_clean, y_pred_clean)[0] if len(y_true_clean) > 1 else np.nan,
        'spearman_rho': spearmanr(y_true_clean, y_pred_clean)[0] if len(y_true_clean) > 1 else np.nan,
        'n_samples': len(y_true_clean)
    }
    return metrics


def plot_scatter(y_true, y_pred, title, ax, color='blue', alpha=0.6):
    """Create scatter plot of predicted vs true values."""
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true_clean = y_true[mask]
    y_pred_clean = y_pred[mask]
    
    ax.scatter(y_true_clean, y_pred_clean, alpha=alpha, color=color, s=20)
    
    # Add diagonal line (perfect predictions)
    min_val = min(y_true_clean.min(), y_pred_clean.min())
    max_val = max(y_true_clean.max(), y_pred_clean.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect prediction')
    
    # Calculate and display metrics
    metrics = calculate_metrics(y_true_clean, y_pred_clean)
    if metrics:
        textstr = f"R² = {metrics['r2']:.4f}\n"
        textstr += f"RMSE = {metrics['rmse']:.4f}\n"
        textstr += f"MAE = {metrics['mae']:.4f}\n"
        textstr += f"Pearson r = {metrics['pearson_r']:.4f}"
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    ax.set_xlabel('True Values', fontsize=12)
    ax.set_ylabel('Predicted Values', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)


def plot_residuals(y_true, y_pred, title, ax, color='blue', alpha=0.6):
    """Create residual plot."""
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true_clean = y_true[mask]
    y_pred_clean = y_pred[mask]
    residuals = y_pred_clean - y_true_clean
    
    ax.scatter(y_true_clean, residuals, alpha=alpha, color=color, s=20)
    ax.axhline(y=0, color='r', linestyle='--', lw=2)
    
    ax.set_xlabel('True Values', fontsize=12)
    ax.set_ylabel('Residuals (Predicted - True)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Add statistics
    mean_residual = np.mean(residuals)
    std_residual = np.std(residuals)
    textstr = f"Mean residual: {mean_residual:.4f}\n"
    textstr += f"Std residual: {std_residual:.4f}"
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))


def plot_distribution(y_true, y_pred, title, ax):
    """Plot distribution of true and predicted values."""
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true_clean = y_true[mask]
    y_pred_clean = y_pred[mask]
    
    ax.hist(y_true_clean, bins=30, alpha=0.6, label='True', color='blue', density=True)
    ax.hist(y_pred_clean, bins=30, alpha=0.6, label='Predicted', color='red', density=True)
    
    ax.set_xlabel('Value', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)


def plot_fold_comparison(df, output_dir):
    """Create comparison plots across folds."""
    n_folds = df['fold'].nunique()
    folds = sorted(df['fold'].unique())
    
    # Metrics per fold
    fold_metrics = []
    for fold in folds:
        fold_data = df[df['fold'] == fold]
        metrics = calculate_metrics(fold_data['y_true'], fold_data['y_pred'])
        if metrics:
            metrics['fold'] = fold
            fold_metrics.append(metrics)
    
    metrics_df = pd.DataFrame(fold_metrics)
    
    # Plot metrics comparison
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Cross-Validation Metrics by Fold', fontsize=16, fontweight='bold')
    
    metrics_to_plot = ['r2', 'rmse', 'mae', 'pearson_r', 'spearman_rho']
    for idx, metric in enumerate(metrics_to_plot):
        ax = axes[idx // 3, idx % 3]
        ax.bar(metrics_df['fold'], metrics_df[metric], color='steelblue', alpha=0.7)
        ax.set_xlabel('Fold', fontsize=11)
        ax.set_ylabel(metric.upper().replace('_', ' '), fontsize=11)
        ax.set_title(f'{metric.upper().replace("_", " ")} by Fold', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_xticks(metrics_df['fold'])
        
        # Add value labels on bars
        for i, v in enumerate(metrics_df[metric]):
            ax.text(metrics_df['fold'].iloc[i], v, f'{v:.4f}', 
                   ha='center', va='bottom', fontsize=9)
    
    # Remove empty subplot
    fig.delaxes(axes[1, 2])
    
    plt.tight_layout()
    metrics_comparison_path = os.path.join(output_dir, 'cv_fold_metrics_comparison.png')
    plt.savefig(metrics_comparison_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {metrics_comparison_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Plot cross-validation predictions')
    parser.add_argument(
        '--results_dir',
        type=str,
        default='peptide_full_no_testset',
        help='Results directory containing CV prediction files'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Output directory for plots (default: same as results_dir)'
    )
    
    args = parser.parse_args()
    
    results_dir = args.results_dir
    output_dir = args.output_dir if args.output_dir else results_dir
    
    print("=" * 80)
    print("CV PREDICTION PLOTTING")
    print("=" * 80)
    print(f"Results directory: {results_dir}")
    print(f"Output directory: {output_dir}")
    print("=" * 80)
    
    # Load combined predictions
    combined_path = os.path.join(results_dir, 'cv_all_folds_predictions.csv')
    if not os.path.exists(combined_path):
        print(f"Error: Combined predictions file not found: {combined_path}")
        return
    
    df = pd.read_csv(combined_path)
    print(f"\nLoaded {len(df)} predictions from {df['fold'].nunique()} folds")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot 1: Combined scatter plot
    print("\n1. Creating combined scatter plot...")
    fig, ax = plt.subplots(figsize=(10, 10))
    plot_scatter(df['y_true'], df['y_pred'], 
                'Cross-Validation Predictions (All Folds)', ax, color='steelblue')
    plt.tight_layout()
    scatter_path = os.path.join(output_dir, 'cv_scatter_all_folds.png')
    plt.savefig(scatter_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {scatter_path}")
    plt.close()
    
    # Plot 2: Combined residual plot
    print("\n2. Creating combined residual plot...")
    fig, ax = plt.subplots(figsize=(10, 8))
    plot_residuals(df['y_true'], df['y_pred'], 
                  'Residuals Plot (All Folds)', ax, color='steelblue')
    plt.tight_layout()
    residual_path = os.path.join(output_dir, 'cv_residuals_all_folds.png')
    plt.savefig(residual_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {residual_path}")
    plt.close()
    
    # Plot 3: Distribution comparison
    print("\n3. Creating distribution plot...")
    fig, ax = plt.subplots(figsize=(10, 8))
    plot_distribution(df['y_true'], df['y_pred'], 
                     'Distribution: True vs Predicted (All Folds)', ax)
    plt.tight_layout()
    dist_path = os.path.join(output_dir, 'cv_distribution_all_folds.png')
    plt.savefig(dist_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {dist_path}")
    plt.close()
    
    # Plot 4: Individual fold scatter plots
    print("\n4. Creating individual fold scatter plots...")
    n_folds = df['fold'].nunique()
    folds = sorted(df['fold'].unique())
    
    n_cols = min(3, n_folds)
    n_rows = (n_folds + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 6*n_rows))
    fig.suptitle('Cross-Validation Predictions by Fold', fontsize=16, fontweight='bold')
    
    if n_folds == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    colors = plt.cm.tab10(np.linspace(0, 1, n_folds))
    
    for idx, fold in enumerate(folds):
        fold_data = df[df['fold'] == fold]
        ax = axes[idx]
        plot_scatter(fold_data['y_true'], fold_data['y_pred'],
                    f'Fold {fold}', ax, color=colors[idx], alpha=0.7)
    
    # Hide extra subplots
    for idx in range(n_folds, len(axes)):
        fig.delaxes(axes[idx])
    
    plt.tight_layout()
    folds_scatter_path = os.path.join(output_dir, 'cv_scatter_by_fold.png')
    plt.savefig(folds_scatter_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {folds_scatter_path}")
    plt.close()
    
    # Plot 5: Fold metrics comparison
    print("\n5. Creating fold metrics comparison...")
    plot_fold_comparison(df, output_dir)
    
    # Calculate and print summary metrics
    print("\n" + "=" * 80)
    print("SUMMARY METRICS")
    print("=" * 80)
    overall_metrics = calculate_metrics(df['y_true'], df['y_pred'])
    if overall_metrics:
        print(f"Overall (All Folds Combined):")
        print(f"  R²:              {overall_metrics['r2']:.4f}")
        print(f"  RMSE:            {overall_metrics['rmse']:.4f}")
        print(f"  MAE:             {overall_metrics['mae']:.4f}")
        print(f"  Pearson r:       {overall_metrics['pearson_r']:.4f}")
        print(f"  Spearman ρ:      {overall_metrics['spearman_rho']:.4f}")
        print(f"  N samples:       {overall_metrics['n_samples']}")
    
    print(f"\nPer-Fold Metrics:")
    for fold in sorted(df['fold'].unique()):
        fold_data = df[df['fold'] == fold]
        fold_metrics = calculate_metrics(fold_data['y_true'], fold_data['y_pred'])
        if fold_metrics:
            print(f"  Fold {fold}: R²={fold_metrics['r2']:.4f}, "
                  f"RMSE={fold_metrics['rmse']:.4f}, "
                  f"Pearson r={fold_metrics['pearson_r']:.4f}")
    
    print("\n" + "=" * 80)
    print("✓ All plots created successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()

