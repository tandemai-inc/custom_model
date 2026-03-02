"""
Compare metrics between two trained models.
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300


def load_training_results(filepath):
    """Load training results JSON file."""
    with open(filepath, 'r') as f:
        results = json.load(f)
    return results


def extract_metrics(results):
    """Extract relevant metrics from training results."""
    metrics = {}
    
    # Basic info
    metrics['model_name'] = results.get('data_file', 'unknown')
    metrics['task'] = results.get('task', 'unknown')
    metrics['n_samples'] = results.get('n_samples', 0)
    metrics['n_features'] = results.get('n_features', 0)
    metrics['feature_selection'] = results.get('feature_selection_used', False)
    
    # CV metrics
    cv_metrics = results.get('cv_metrics', {})
    
    if results.get('task') == 'regression':
        metrics['cv_r2_mean'] = cv_metrics.get('r2_mean', None)
        metrics['cv_r2_std'] = cv_metrics.get('r2_std', None)
        metrics['cv_mae_mean'] = cv_metrics.get('mae_mean', None)
        metrics['cv_mae_std'] = cv_metrics.get('mae_std', None)
        metrics['cv_rmse_mean'] = cv_metrics.get('rmse_mean', None)
        metrics['cv_rmse_std'] = cv_metrics.get('rmse_std', None)
        metrics['cv_pearson_r_mean'] = cv_metrics.get('pearson_r_mean', None)
        metrics['cv_pearson_r_std'] = cv_metrics.get('pearson_r_std', None)
        metrics['cv_spearman_rho_mean'] = cv_metrics.get('spearman_rho_mean', None)
        metrics['cv_spearman_rho_std'] = cv_metrics.get('spearman_rho_std', None)
    else:
        # Classification metrics
        metrics['cv_accuracy_mean'] = cv_metrics.get('accuracy_mean', None)
        metrics['cv_f1_mean'] = cv_metrics.get('f1_mean', None)
        metrics['cv_roc_auc_mean'] = cv_metrics.get('roc_auc_mean', None)
        metrics['cv_pr_auc_mean'] = cv_metrics.get('pr_auc_mean', None)
    
    # Test metrics (if available)
    test_metrics = results.get('test_metrics', {})
    if test_metrics:
        if results.get('task') == 'regression':
            metrics['test_r2'] = test_metrics.get('r2', None)
            metrics['test_mae'] = test_metrics.get('mae', None)
            metrics['test_rmse'] = test_metrics.get('rmse', None)
            metrics['test_pearson_r'] = test_metrics.get('pearson_r', None)
            metrics['test_spearman_rho'] = test_metrics.get('spearman_rho', None)
        else:
            metrics['test_accuracy'] = test_metrics.get('accuracy', None)
            metrics['test_f1'] = test_metrics.get('f1', None)
            metrics['test_roc_auc'] = test_metrics.get('roc_auc', None)
    
    return metrics


def create_comparison_table(metrics_list, model_names):
    """Create a comparison DataFrame."""
    df = pd.DataFrame(metrics_list)
    df.index = model_names
    return df


def plot_metric_comparison(metrics_list, model_names, output_path=None, task='regression'):
    """Create visualization comparing metrics."""
    if task == 'regression':
        # Regression metrics
        metric_pairs = [
            ('cv_r2_mean', 'cv_r2_std', 'R²'),
            ('cv_mae_mean', 'cv_mae_std', 'MAE'),
            ('cv_rmse_mean', 'cv_rmse_std', 'RMSE'),
            ('cv_pearson_r_mean', 'cv_pearson_r_std', 'Pearson r'),
            ('cv_spearman_rho_mean', 'cv_spearman_rho_std', 'Spearman ρ')
        ]
    else:
        # Classification metrics
        metric_pairs = [
            ('cv_accuracy_mean', None, 'Accuracy'),
            ('cv_f1_mean', None, 'F1 Score'),
            ('cv_roc_auc_mean', None, 'ROC-AUC'),
            ('cv_pr_auc_mean', None, 'PR-AUC')
        ]
    
    n_metrics = len(metric_pairs)
    fig, axes = plt.subplots(1, n_metrics, figsize=(5*n_metrics, 6))
    
    if n_metrics == 1:
        axes = [axes]
    
    for idx, (mean_key, std_key, metric_name) in enumerate(metric_pairs):
        ax = axes[idx]
        
        means = []
        stds = []
        labels = []
        
        for i, metrics in enumerate(metrics_list):
            mean_val = metrics.get(mean_key)
            std_val = metrics.get(std_key) if std_key else None
            
            if mean_val is not None:
                means.append(mean_val)
                stds.append(std_val if std_val is not None else 0)
                labels.append(model_names[i])
        
        if means:
            x_pos = range(len(means))
            bars = ax.bar(x_pos, means, yerr=stds if any(s is not None for s in stds) else None,
                         capsize=5, alpha=0.8, color=['#3498DB', '#E74C3C'][:len(means)])
            
            ax.set_xticks(x_pos)
            ax.set_xticklabels(labels, rotation=45, ha='right')
            ax.set_ylabel(metric_name, fontsize=11, fontweight='bold')
            ax.set_title(f'{metric_name} Comparison', fontsize=12, fontweight='bold')
            ax.grid(axis='y', alpha=0.3, linestyle='--')
            
            # Add value labels on bars
            for i, (bar, mean, std) in enumerate(zip(bars, means, stds)):
                height = bar.get_height()
                label = f'{mean:.4f}'
                if std is not None and std > 0:
                    label += f' ± {std:.4f}'
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       label, ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, bbox_inches='tight', facecolor='white')
        print(f"Comparison plot saved to {output_path}")
    
    return fig, axes


def print_comparison_table(df):
    """Print a formatted comparison table."""
    print("\n" + "=" * 100)
    print("MODEL COMPARISON")
    print("=" * 100)
    
    # Basic info
    print("\nBasic Information:")
    print("-" * 100)
    basic_cols = ['n_samples', 'n_features', 'feature_selection', 'task']
    if all(col in df.columns for col in basic_cols):
        print(df[basic_cols].to_string())
    
    # CV metrics
    if df['task'].iloc[0] == 'regression':
        print("\nCross-Validation Metrics (Mean ± Std):")
        print("-" * 100)
        cv_cols = ['cv_r2_mean', 'cv_mae_mean', 'cv_rmse_mean', 
                   'cv_pearson_r_mean', 'cv_spearman_rho_mean']
        cv_df = df[cv_cols].copy()
        
        # Format with std
        for col in cv_cols:
            mean_col = col
            std_col = col.replace('_mean', '_std')
            if std_col in df.columns:
                cv_df[col] = df.apply(
                    lambda row: f"{row[mean_col]:.4f} ± {row[std_col]:.4f}" 
                    if pd.notna(row[mean_col]) and pd.notna(row[std_col]) 
                    else f"{row[mean_col]:.4f}" if pd.notna(row[mean_col]) else "N/A",
                    axis=1
                )
        
        print(cv_df.to_string())
        
        # Test metrics if available
        test_cols = ['test_r2', 'test_mae', 'test_rmse', 'test_pearson_r', 'test_spearman_rho']
        if any(col in df.columns for col in test_cols):
            print("\nTest Set Metrics:")
            print("-" * 100)
            test_df = df[[col for col in test_cols if col in df.columns]]
            print(test_df.to_string())
    else:
        print("\nCross-Validation Metrics:")
        print("-" * 100)
        cv_cols = ['cv_accuracy_mean', 'cv_f1_mean', 'cv_roc_auc_mean', 'cv_pr_auc_mean']
        cv_df = df[[col for col in cv_cols if col in df.columns]]
        print(cv_df.to_string())
    
    print("\n" + "=" * 100)


def main():
    parser = argparse.ArgumentParser(description='Compare metrics between two trained models')
    parser.add_argument('--model1', type=str, required=True,
                       help='Path to first model training_results.json')
    parser.add_argument('--model2', type=str, required=True,
                       help='Path to second model training_results.json')
    parser.add_argument('--name1', type=str, default=None,
                       help='Name for first model (default: inferred from path)')
    parser.add_argument('--name2', type=str, default=None,
                       help='Name for second model (default: inferred from path)')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory for comparison files (default: current directory)')
    
    args = parser.parse_args()
    
    # Load results
    print(f"Loading model 1: {args.model1}")
    results1 = load_training_results(args.model1)
    metrics1 = extract_metrics(results1)
    
    print(f"Loading model 2: {args.model2}")
    results2 = load_training_results(args.model2)
    metrics2 = extract_metrics(results2)
    
    # Get model names
    if args.name1:
        name1 = args.name1
    else:
        name1 = Path(args.model1).parent.name
        if name1 == 'all_results':
            name1 = Path(args.model1).parent.parent.name
    
    if args.name2:
        name2 = args.name2
    else:
        name2 = Path(args.model2).parent.name
        if name2 == 'all_results':
            name2 = Path(args.model2).parent.parent.name
    
    # Create comparison
    metrics_list = [metrics1, metrics2]
    model_names = [name1, name2]
    
    df = create_comparison_table(metrics_list, model_names)
    
    # Print comparison
    print_comparison_table(df)
    
    # Save comparison table
    output_dir = args.output_dir if args.output_dir else os.getcwd()
    os.makedirs(output_dir, exist_ok=True)
    
    csv_path = os.path.join(output_dir, 'model_comparison.csv')
    df.to_csv(csv_path)
    print(f"\nComparison table saved to {csv_path}")
    
    # Create visualization
    task = metrics1.get('task', 'regression')
    plot_path = os.path.join(output_dir, 'model_comparison_plot.png')
    plot_metric_comparison(metrics_list, model_names, output_path=plot_path, task=task)
    
    print("\nDone!")


if __name__ == "__main__":
    main()

