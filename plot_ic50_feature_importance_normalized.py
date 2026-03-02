"""
Plot feature importance with normalized importance values.
Based on user-provided plotting code.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Set style
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300

# Path to feature importance file
work_dir = '/Users/chrishe/Documents/Untitled Folder 3/xgboost_training/peptide_IC50_with_selection/'
feature_file = os.path.join(work_dir, 'feature_importance.csv')

# Load feature importance
print(f"Loading feature importance from {feature_file}...")
feature_df = pd.read_csv(feature_file)
print(f"  Loaded {len(feature_df)} features")

# Normalize the importance to percentages
feature_df['importance'] = feature_df['importance'] * 100 / feature_df['importance'].sum()
print(f"  Normalized importance (sum = {feature_df['importance'].sum():.2f}%)")

# Get top 300 features
feature_df = feature_df.head(300)
print(f"  Using top {len(feature_df)} features")

# Assign colors based on feature type
colors = np.where(feature_df['feature'].str.contains('embedding', case=False, na=False),
                  'tab:red', 'tab:blue')

# Count features and calculate cumulative importance
is_pepland = colors == 'tab:red'
num_pepland = is_pepland.sum()
num_other = (~is_pepland).sum()
pepland_importance = feature_df.loc[is_pepland, 'importance'].sum()
other_importance = feature_df.loc[~is_pepland, 'importance'].sum()

print(f"\nFeature Statistics (Top 300):")
print(f"  Embedding features: {num_pepland} ({pepland_importance:.2f}% of total importance)")
print(f"  Other features: {num_other} ({other_importance:.2f}% of total importance)")

# Create the plot
plt.figure(figsize=(10, 6))
bars = plt.bar(feature_df['feature'], feature_df['importance'], color=colors)

# Customize the plot
plt.xlabel('Feature', fontsize=12, fontweight='bold')
plt.ylabel('Importance (%)', fontsize=12, fontweight='bold')
plt.title(f'Top 300 Feature Importances - Peptide IC50 Model\n(Embeddings: {num_pepland}, Other: {num_other})', 
          fontsize=14, fontweight='bold', pad=15)

# Remove x-axis tick labels
plt.xticks([])
plt.tight_layout()

# Add legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='tab:red', label=f'Embedding ({num_pepland}, {pepland_importance:.2f}%)'),
    Patch(facecolor='tab:blue', label=f'Other ({num_other}, {other_importance:.2f}%)')
]
plt.legend(handles=legend_elements, loc='upper right', fontsize=10)

# Add grid
plt.grid(axis='y', alpha=0.3, linestyle='--')

# Save the plot
output_path = os.path.expanduser('~/Downloads/feature_importance_normalized_top300.png')
plt.savefig(output_path, bbox_inches='tight', facecolor='white', dpi=300)
print(f"\nPlot saved to {output_path}")
plt.close()

print("\nDone!")

