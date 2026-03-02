"""
Plot CV prediction results using custom plotting function.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr
import argparse
import os


def plot_prediction_results(labels, predictions, x_label, y_label, title, output_fig):
    """Plot prediction results with metrics."""
    rmse_all = np.sqrt(np.mean((labels - predictions) ** 2))
    
    spearman_rho, _ = spearmanr(labels, predictions)
    pearson_r, _ = pearsonr(labels, predictions)
    
    # Create scatter plot
    plt.figure(figsize=(8, 8))
    plt.scatter(labels, predictions, alpha=0.5, s=20)
    plt.plot([labels.min(), labels.max()], [labels.min(), labels.max()], 'r--', lw=2)
    plt.xlabel(x_label, fontsize=14)
    plt.ylabel(y_label, fontsize=14)
    # increase tick label size
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.title(title, fontsize=15)
    
    # Add metrics as legend
    legend_text = f"RMSE: {rmse_all:.4f}\nSpearman's ρ: {spearman_rho:.4f}\nPearson's r: {pearson_r:.4f}"
    plt.text(0.05, 0.95, legend_text, transform=plt.gca().transAxes, 
             fontsize=14, verticalalignment='top', 
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_fig, dpi=300)
    plt.close()  # Close instead of show for script usage
    
    print(f"Total test samples: {len(predictions)}")
    print(f"RMSE: {rmse_all:.6f}")
    print(f"Spearman's ρ: {spearman_rho:.6f}")
    print(f"Pearson's r: {pearson_r:.6f}")


def main():
    parser = argparse.ArgumentParser(description='Plot CV prediction results')
    parser.add_argument('--cv_predictions', type=str, required=True,
                       help='Path to CV predictions CSV file (cv_all_folds_predictions.csv)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output figure path (default: same directory as input with .png extension)')
    parser.add_argument('--x_label', type=str, default='True Label',
                       help='X-axis label (default: True Label)')
    parser.add_argument('--y_label', type=str, default='Predicted Label',
                       help='Y-axis label (default: Predicted Label)')
    parser.add_argument('--title', type=str, default=None,
                       help='Plot title (default: inferred from file path)')
    
    args = parser.parse_args()
    
    # Load CV predictions
    print(f"Loading CV predictions from {args.cv_predictions}...")
    df = pd.read_csv(args.cv_predictions)
    
    # Check for required columns (try multiple common naming conventions)
    if 'y_true' in df.columns and 'y_pred' in df.columns:
        labels = df['y_true'].values
        predictions = df['y_pred'].values
    elif 'true_label' in df.columns and 'prediction' in df.columns:
        labels = df['true_label'].values
        predictions = df['prediction'].values
    elif 'label' in df.columns and 'predicted' in df.columns:
        labels = df['label'].values
        predictions = df['predicted'].values
    else:
        raise ValueError(f"Could not find label/prediction columns. Available columns: {list(df.columns)}")
    
    print(f"  Loaded {len(labels)} predictions")
    
    # Determine output path
    if args.output is None:
        base_path = os.path.splitext(args.cv_predictions)[0]
        output_path = f"{base_path}_plot.png"
    else:
        output_path = args.output
    
    # Determine title
    if args.title is None:
        # Infer from file path
        dir_name = os.path.basename(os.path.dirname(args.cv_predictions))
        title = f"Cross-Validation Predictions: {dir_name}"
    else:
        title = args.title
    
    # Create plot
    print(f"\nCreating plot...")
    plot_prediction_results(
        labels, predictions,
        x_label=args.x_label,
        y_label=args.y_label,
        title=title,
        output_fig=output_path
    )
    
    print(f"\nPlot saved to {output_path}")


if __name__ == "__main__":
    main()

