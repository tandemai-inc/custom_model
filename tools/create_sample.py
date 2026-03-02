"""Create a small sample from the peptide data for testing."""

import pandas as pd
import os

file_path = os.path.expanduser('~/Downloads/processed_data_filtered_with_embeddings.csv')
output_path = os.path.join(os.path.dirname(__file__), 'peptide_sample_20.csv')

print(f"Loading from: {file_path}")
df = pd.read_csv(file_path)

print(f"Full dataset: {len(df)} samples, {len(df.columns)} columns")

# Create a small sample (first 20 samples)
sample_df = df.head(20).copy()
sample_df.to_csv(output_path, index=False)

print(f"\nCreated sample file: {output_path}")
print(f"Sample size: {len(sample_df)} samples")
print(f"\nColumns:")
non_embed_cols = [col for col in sample_df.columns if not col.startswith('embedding_')]
print(f"  Metadata: {len(non_embed_cols)} columns")
print(f"  Embeddings: {sum(1 for col in sample_df.columns if col.startswith('embedding_'))} features")
print(f"\nFirst few rows:")
print(sample_df[['smiles', 'label']].head())

