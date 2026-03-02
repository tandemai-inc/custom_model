"""
Plot feature importance with highlighting for embedding (pepland) features.
"""

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


def load_feature_importance(filepath):
    """Load feature importance CSV file."""
    df = pd.read_csv(filepath)
    if 'feature' not in df.columns or 'importance' not in df.columns:
        raise ValueError(f"Expected columns 'feature' and 'importance' in {filepath}")
    return df


def identify_feature_type(feature_name):
    """Identify if a feature is an embedding (pepland) or molecular feature."""
    if feature_name.startswith('embedding_'):
        return 'Embedding (PepLand)'
    elif feature_name.startswith('mordred_'):
        return 'Mordred'
    elif feature_name.startswith('morgan_'):
        return 'Morgan'
    elif feature_name.startswith('rdkit_'):
        return 'RDKit'
    elif feature_name.startswith('maccs_'):
        return 'MACCS'
    elif feature_name.startswith('map4_'):
        return 'MAP4'
    else:
        return 'Other'


def plot_feature_importance(df, output_path=None, top_n=50, figsize=(12, 8)):
    """
    Plot feature importance with color coding for embedding features.
    
    Args:
        df: DataFrame with 'feature' and 'importance' columns
        output_path: Path to save the plot (optional)
        top_n: Number of top features to plot (default: 50)
        figsize: Figure size tuple (default: (12, 8))
    """
    # Sort by importance
    df_sorted = df.sort_values('importance', ascending=False).head(top_n)
    
    # Identify feature types
    df_sorted['feature_type'] = df_sorted['feature'].apply(identify_feature_type)
    
    # Create color mapping
    color_map = {
        'Embedding (PepLand)': '#E74C3C',  # Red for embeddings
        'Mordred': '#3498DB',              # Blue for Mordred
        'Morgan': '#2ECC71',               # Green for Morgan
        'RDKit': '#9B59B6',                # Purple for RDKit
        'MACCS': '#F39C12',                # Orange for MACCS
        'MAP4': '#1ABC9C',                 # Teal for MAP4
        'Other': '#95A5A6'                 # Gray for other
    }
    
    # Get colors for each feature
    colors = [color_map.get(ftype, '#95A5A6') for ftype in df_sorted['feature_type']]
    
    # Create the plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Horizontal bar plot
    bars = ax.barh(range(len(df_sorted)), df_sorted['importance'], color=colors)
    
    # Set y-axis labels
    ax.set_yticks(range(len(df_sorted)))
    ax.set_yticklabels(df_sorted['feature'], fontsize=9)
    
    # Labels and title
    ax.set_xlabel('Feature Importance', fontsize=12, fontweight='bold')
    ax.set_ylabel('Feature', fontsize=12, fontweight='bold')
    ax.set_title(f'Top {top_n} Feature Importances\n(Highlighted: Embedding/PepLand Features)', 
                 fontsize=14, fontweight='bold', pad=20)
    
    # Invert y-axis to show highest importance at top
    ax.invert_yaxis()
    
    # Add grid
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    
    # Create legend
    legend_elements = []
    for ftype, color in color_map.items():
        count = (df_sorted['feature_type'] == ftype).sum()
        if count > 0:
            legend_elements.append(plt.Rectangle((0, 0), 1, 1, facecolor=color, 
                                                  label=f'{ftype} ({count})'))
    
    ax.legend(handles=legend_elements, loc='lower right', fontsize=10)
    
    # Tight layout
    plt.tight_layout()
    
    # Save if output path provided
    if output_path:
        plt.savefig(output_path, bbox_inches='tight', facecolor='white')
        print(f"Plot saved to {output_path}")
    
    return fig, ax


def plot_feature_importance_distribution(df, output_path=None, figsize=(14, 6)):
    """
    Create a side-by-side comparison: bar plot and distribution by feature type.
    
    Args:
        df: DataFrame with 'feature' and 'importance' columns
        output_path: Path to save the plot (optional)
        figsize: Figure size tuple (default: (14, 6))
    """
    # Identify feature types
    df['feature_type'] = df['feature'].apply(identify_feature_type)
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Plot 1: Top features bar plot
    top_n = 30
    df_sorted = df.sort_values('importance', ascending=False).head(top_n)
    
    color_map = {
        'Embedding (PepLand)': '#E74C3C',
        'Mordred': '#3498DB',
        'Morgan': '#2ECC71',
        'RDKit': '#9B59B6',
        'MACCS': '#F39C12',
        'MAP4': '#1ABC9C',
        'Other': '#95A5A6'
    }
    
    colors = [color_map.get(ftype, '#95A5A6') for ftype in df_sorted['feature_type']]
    
    axes[0].barh(range(len(df_sorted)), df_sorted['importance'], color=colors)
    axes[0].set_yticks(range(len(df_sorted)))
    axes[0].set_yticklabels(df_sorted['feature'], fontsize=8)
    axes[0].set_xlabel('Feature Importance', fontsize=11, fontweight='bold')
    axes[0].set_ylabel('Feature', fontsize=11, fontweight='bold')
    axes[0].set_title(f'Top {top_n} Features', fontsize=12, fontweight='bold')
    axes[0].invert_yaxis()
    axes[0].grid(axis='x', alpha=0.3, linestyle='--')
    
    # Plot 2: Distribution by feature type
    feature_type_stats = df.groupby('feature_type')['importance'].agg(['mean', 'std', 'count'])
    feature_type_stats = feature_type_stats.sort_values('mean', ascending=False)
    
    bars = axes[1].bar(range(len(feature_type_stats)), feature_type_stats['mean'],
                      color=[color_map.get(ftype, '#95A5A6') for ftype in feature_type_stats.index],
                      yerr=feature_type_stats['std'], capsize=5, alpha=0.8)
    
    axes[1].set_xticks(range(len(feature_type_stats)))
    axes[1].set_xticklabels(feature_type_stats.index, rotation=45, ha='right', fontsize=10)
    axes[1].set_ylabel('Mean Importance', fontsize=11, fontweight='bold')
    axes[1].set_title('Mean Importance by Feature Type', fontsize=12, fontweight='bold')
    axes[1].grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add count labels on bars
    for i, (idx, row) in enumerate(feature_type_stats.iterrows()):
        axes[1].text(i, row['mean'] + row['std'] + 0.01, f'n={int(row["count"])}',
                    ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, bbox_inches='tight', facecolor='white')
        print(f"Plot saved to {output_path}")
    
    return fig, axes


def main():
    parser = argparse.ArgumentParser(description='Plot feature importance with embedding highlights')
    parser.add_argument('--importance_file', type=str, required=True,
                       help='Path to feature importance CSV file')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory for plots (default: same as importance file)')
    parser.add_argument('--top_n', type=int, default=50,
                       help='Number of top features to plot (default: 50)')
    parser.add_argument('--create_distribution', action='store_true',
                       help='Also create a distribution plot by feature type')
    
    args = parser.parse_args()
    
    # Load feature importance
    print(f"Loading feature importance from {args.importance_file}...")
    df = load_feature_importance(args.importance_file)
    print(f"  Loaded {len(df)} features")
    
    # Identify feature types
    df['feature_type'] = df['feature'].apply(identify_feature_type)
    
    # Print summary
    print("\nFeature Type Summary:")
    summary = df.groupby('feature_type').agg({
        'importance': ['count', 'mean', 'sum']
    }).round(4)
    summary.columns = ['Count', 'Mean Importance', 'Total Importance']
    print(summary)
    
    # Determine output directory
    if args.output_dir is None:
        output_dir = os.path.dirname(args.importance_file)
    else:
        output_dir = args.output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    # Create main plot
    print(f"\nCreating feature importance plot (top {args.top_n})...")
    output_path = os.path.join(output_dir, 'feature_importance_plot.png')
    plot_feature_importance(df, output_path=output_path, top_n=args.top_n)
    
    # Create distribution plot if requested
    if args.create_distribution:
        print("\nCreating distribution plot...")
        dist_output_path = os.path.join(output_dir, 'feature_importance_distribution.png')
        plot_feature_importance_distribution(df, output_path=dist_output_path)
    
    print("\nDone!")


if __name__ == "__main__":
    main()

