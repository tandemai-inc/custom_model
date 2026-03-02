"""
Plot feature importance for IC50 model with embedding highlights.
Highlights embedding features in a different color.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os

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
    """Identify if a feature is an embedding or other type."""
    if feature_name.startswith('embedding_'):
        return 'Embedding'
    else:
        return 'Other'


def plot_feature_importance(df, output_path, top_n=100, figsize=(12, 10)):
    """
    Plot feature importance with color coding for embedding features.
    
    Args:
        df: DataFrame with 'feature' and 'importance' columns
        output_path: Path to save the plot
        top_n: Number of top features to plot
        figsize: Figure size tuple
    """
    # Sort by importance and get top N
    df_sorted = df.sort_values('importance', ascending=False).head(top_n)
    
    # Identify feature types
    df_sorted['feature_type'] = df_sorted['feature'].apply(identify_feature_type)
    
    # Create color mapping
    color_map = {
        'Embedding': '#E74C3C',  # Red for embeddings
        'Other': '#3498DB'        # Blue for other features
    }
    
    # Get colors for each feature
    colors = [color_map.get(ftype, '#95A5A6') for ftype in df_sorted['feature_type']]
    
    # Count features by type
    embedding_count = (df_sorted['feature_type'] == 'Embedding').sum()
    other_count = (df_sorted['feature_type'] == 'Other').sum()
    
    # Create the plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Horizontal bar plot
    bars = ax.barh(range(len(df_sorted)), df_sorted['importance'], color=colors)
    
    # Set y-axis labels
    ax.set_yticks(range(len(df_sorted)))
    ax.set_yticklabels(df_sorted['feature'], fontsize=9)
    
    # Labels and title
    ax.set_xlabel('Feature Importance', fontsize=14, fontweight='bold')
    ax.set_ylabel('Feature', fontsize=14, fontweight='bold')
    ax.set_title(f'Top {top_n} Feature Importances - Peptide IC50 Model\n(Embeddings: {embedding_count}, Other: {other_count})', 
                 fontsize=16, fontweight='bold', pad=20)
    
    # Invert y-axis to show highest importance at top
    ax.invert_yaxis()
    
    # Add grid
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    
    # Create legend
    from matplotlib.patches import Rectangle
    legend_elements = [
        Rectangle((0, 0), 1, 1, facecolor='#E74C3C', label=f'Embedding ({embedding_count})'),
        Rectangle((0, 0), 1, 1, facecolor='#3498DB', label=f'Other ({other_count})')
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=12)
    
    # Tight layout
    plt.tight_layout()
    
    # Save
    plt.savefig(output_path, bbox_inches='tight', facecolor='white')
    print(f"Plot saved to {output_path}")
    plt.close()
    
    return fig


def main():
    parser = argparse.ArgumentParser(description='Plot IC50 feature importance with embedding highlights')
    parser.add_argument('--importance_file', type=str, required=True,
                       help='Path to feature importance CSV file')
    parser.add_argument('--output_dir', type=str, default=os.path.expanduser('~/Downloads'),
                       help='Output directory for plots (default: ~/Downloads)')
    parser.add_argument('--top_n_list', type=str, default='100,200,300,500',
                       help='Comma-separated list of top N values to plot (default: 100,200,300,500)')
    
    args = parser.parse_args()
    
    # Parse top N list
    top_n_values = [int(x.strip()) for x in args.top_n_list.split(',')]
    
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
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create plots for each top N value
    base_name = os.path.splitext(os.path.basename(args.importance_file))[0]
    
    for top_n in top_n_values:
        if top_n > len(df):
            print(f"\nSkipping top_{top_n} (only {len(df)} features available)")
            continue
        
        print(f"\nCreating plot for top {top_n} features...")
        output_path = os.path.join(args.output_dir, f'{base_name}_top{top_n}.png')
        plot_feature_importance(df, output_path, top_n=top_n)
    
    print("\n" + "=" * 80)
    print("All plots created successfully!")
    print(f"Plots saved to: {args.output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()

