"""
Script to concatenate peptide embeddings with molecular features for regression training.

This script:
1. Loads peptide data with pre-computed embeddings
2. Calculates molecular features from SMILES strings
3. Concatenates embeddings with molecular features
4. Saves the combined dataset for training
"""

import pandas as pd
import numpy as np
import argparse
import os
import sys
from pathlib import Path

# Add current directory to path to import local modules
sys.path.insert(0, str(Path(__file__).parent))

from featurization import calculate_all_features


def load_peptide_data(filepath):
    """Load peptide data with embeddings."""
    print(f"Loading peptide data from {filepath}...")
    df = pd.read_csv(filepath)
    print(f"  Loaded {len(df)} samples")
    
    # Check required columns
    required_cols = ['smiles', 'label']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Identify embedding columns
    embedding_cols = [col for col in df.columns if col.startswith('embedding_')]
    print(f"  Found {len(embedding_cols)} embedding features")
    
    # Identify columns to exclude (label-related columns that shouldn't be features)
    exclude_cols = ['label', 'smiles']
    # Also exclude common label-like column names
    label_like_patterns = ['permeability', 'pampa', 'caco2', 'mdck', 'rrck']
    for col in df.columns:
        col_lower = col.lower()
        if any(pattern in col_lower for pattern in label_like_patterns) and col != 'label':
            exclude_cols.append(col)
            print(f"  Warning: Excluding potential label column: {col}")
    
    return df, embedding_cols, exclude_cols


def calculate_molecular_features(smiles_list, reduced_features_path=None, 
                                 include_map4=True, map4_dimensions=1024):
    """Calculate molecular features from SMILES strings."""
    print("\nCalculating molecular features from SMILES...")
    X_molecular = calculate_all_features(
        smiles_list,
        reduced_features_path=reduced_features_path,
        include_map4=include_map4,
        map4_dimensions=map4_dimensions
    )
    print(f"  Calculated {X_molecular.shape[1]} molecular features")
    return X_molecular


def concatenate_features(df_peptide, embedding_cols, X_molecular):
    """Concatenate peptide embeddings with molecular features."""
    print("\nConcatenating features...")
    
    # Extract embeddings
    X_embeddings = df_peptide[embedding_cols].copy()
    print(f"  Embedding features: {X_embeddings.shape[1]}")
    print(f"  Molecular features: {X_molecular.shape[1]}")
    
    # Ensure same number of rows
    if len(X_embeddings) != len(X_molecular):
        raise ValueError(
            f"Mismatch in number of samples: "
            f"embeddings={len(X_embeddings)}, molecular={len(X_molecular)}"
        )
    
    # Reset indices to ensure alignment
    X_embeddings = X_embeddings.reset_index(drop=True)
    X_molecular = X_molecular.reset_index(drop=True)
    
    # Concatenate horizontally
    X_combined = pd.concat([X_embeddings, X_molecular], axis=1)
    print(f"  Combined features: {X_combined.shape[1]} total")
    
    return X_combined


def main():
    parser = argparse.ArgumentParser(
        description='Concatenate peptide embeddings with molecular features'
    )
    parser.add_argument(
        '--peptide_data',
        type=str,
        default=os.path.expanduser('~/Downloads/processed_data_filtered_with_embeddings.csv'),
        help='Path to peptide data CSV file with embeddings'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='peptide_molecular_combined.csv',
        help='Output CSV file path (default: peptide_molecular_combined.csv)'
    )
    parser.add_argument(
        '--reduced_features',
        type=str,
        default='../reduced_mordred_features.json',
        help='Path to reduced Mordred features JSON file'
    )
    parser.add_argument(
        '--include_map4',
        action='store_true',
        default=True,
        help='Include MAP4 fingerprints (default: True)'
    )
    parser.add_argument(
        '--map4_dimensions',
        type=int,
        default=1024,
        help='MAP4 fingerprint dimensions (default: 1024)'
    )
    parser.add_argument(
        '--save_features_only',
        action='store_true',
        help='Save only features (no labels/metadata) for training'
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("Peptide + Molecular Feature Concatenation")
    print("=" * 80)
    print(f"Peptide data: {args.peptide_data}")
    print(f"Output: {args.output}")
    print(f"Reduced features: {args.reduced_features}")
    print("=" * 80)
    
    # Step 1: Load peptide data
    df_peptide, embedding_cols, exclude_cols = load_peptide_data(args.peptide_data)
    
    # Step 2: Calculate molecular features
    # Filter out rows with missing SMILES
    df_valid = df_peptide.dropna(subset=['smiles']).reset_index(drop=True)
    smiles_list = df_valid['smiles'].tolist()
    
    print(f"  Valid samples (with SMILES): {len(df_valid)}")
    
    X_molecular = calculate_molecular_features(
        smiles_list,
        reduced_features_path=args.reduced_features if os.path.exists(args.reduced_features) else None,
        include_map4=args.include_map4,
        map4_dimensions=args.map4_dimensions
    )
    
    # Step 3: Concatenate features
    X_combined = concatenate_features(df_valid, embedding_cols, X_molecular)
    
    # Step 4: Prepare output
    if args.save_features_only:
        # Save only features (for training)
        output_df = X_combined.copy()
        print(f"\nSaving features only to {args.output}...")
    else:
        # Save with labels and metadata
        # Get metadata columns (everything except embeddings and excluded columns)
        # Keep only essential metadata: smiles, label, and optionally fold
        essential_metadata = ['smiles', 'label']
        if 'fold' in df_valid.columns:
            essential_metadata.append('fold')
        
        # Get other metadata columns (excluding label-like columns)
        other_metadata = [col for col in df_valid.columns 
                         if col not in embedding_cols 
                         and col not in exclude_cols
                         and col not in essential_metadata]
        
        # Combine: essential metadata + other safe metadata + features
        metadata_to_include = essential_metadata + other_metadata
        output_df = pd.concat([
            df_valid[metadata_to_include].reset_index(drop=True),
            X_combined
        ], axis=1)
        print(f"\nSaving combined dataset to {args.output}...")
        print(f"  Included metadata: {len(metadata_to_include)} columns")
        print(f"  Excluded label-like columns: {exclude_cols}")
    
    # Save to CSV
    output_df.to_csv(args.output, index=False)
    print(f"  Saved {len(output_df)} samples with {output_df.shape[1]} columns")
    
    # Print summary
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"Total samples: {len(output_df)}")
    print(f"Total features: {output_df.shape[1]}")
    print(f"  - Embedding features: {len(embedding_cols)}")
    print(f"  - Molecular features: {X_molecular.shape[1]}")
    if not args.save_features_only:
        print(f"  - Metadata columns: {len(output_df.columns) - len(embedding_cols) - X_molecular.shape[1]}")
    print(f"\nOutput saved to: {args.output}")
    print("=" * 80)


if __name__ == "__main__":
    main()

