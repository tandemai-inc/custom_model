"""
Combine features from one file with labels from another file.
"""

import pandas as pd
import argparse
import os


def main():
    parser = argparse.ArgumentParser(
        description='Combine features from one CSV with labels from another CSV'
    )
    parser.add_argument(
        '--features_file',
        type=str,
        required=True,
        help='CSV file containing features (molecular + embeddings)'
    )
    parser.add_argument(
        '--labels_file',
        type=str,
        required=True,
        help='CSV file containing labels'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output CSV file path'
    )
    parser.add_argument(
        '--match_column',
        type=str,
        default='smiles',
        help='Column to match on (default: smiles)'
    )
    parser.add_argument(
        '--label_column',
        type=str,
        default='label',
        help='Label column name in labels file (default: label)'
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("Combining Features and Labels")
    print("=" * 80)
    print(f"Features file: {args.features_file}")
    print(f"Labels file: {args.labels_file}")
    print(f"Output: {args.output}")
    print(f"Match column: {args.match_column}")
    print("=" * 80)
    
    # Load features file
    print(f"\n1. Loading features from {args.features_file}...")
    df_features = pd.read_csv(args.features_file)
    print(f"   Loaded {len(df_features)} samples")
    print(f"   Columns: {len(df_features.columns)}")
    
    # Load labels file
    print(f"\n2. Loading labels from {args.labels_file}...")
    df_labels = pd.read_csv(args.labels_file)
    print(f"   Loaded {len(df_labels)} samples")
    print(f"   Columns: {len(df_labels.columns)}")
    
    # Check if match column exists in both
    if args.match_column not in df_features.columns:
        raise ValueError(f"Match column '{args.match_column}' not found in features file. Available: {list(df_features.columns[:10])}")
    
    if args.match_column not in df_labels.columns:
        raise ValueError(f"Match column '{args.match_column}' not found in labels file. Available: {list(df_labels.columns[:10])}")
    
    if args.label_column not in df_labels.columns:
        raise ValueError(f"Label column '{args.label_column}' not found in labels file. Available: {list(df_labels.columns[:10])}")
    
    # Merge on match column
    print(f"\n3. Merging on '{args.match_column}'...")
    
    # Identify feature columns (everything except match column and label if it exists)
    feature_cols = [col for col in df_features.columns 
                   if col != args.match_column and col != 'label']
    
    # Select only match column and features from features file
    df_features_subset = df_features[[args.match_column] + feature_cols].copy()
    
    # Select only match column and label from labels file
    df_labels_subset = df_labels[[args.match_column, args.label_column]].copy()
    df_labels_subset = df_labels_subset.rename(columns={args.label_column: 'label'})
    
    # Merge
    df_combined = pd.merge(
        df_features_subset,
        df_labels_subset,
        on=args.match_column,
        how='inner'
    )
    
    print(f"   Matched {len(df_combined)} samples")
    print(f"   Features: {len(feature_cols)}")
    
    # Reorder columns: match_column, label, then features
    output_cols = [args.match_column, 'label'] + feature_cols
    df_combined = df_combined[output_cols]
    
    # Save
    print(f"\n4. Saving to {args.output}...")
    df_combined.to_csv(args.output, index=False)
    print(f"   Saved {len(df_combined)} samples with {len(df_combined.columns)} columns")
    
    # Summary
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"Total samples: {len(df_combined)}")
    print(f"Total columns: {len(df_combined.columns)}")
    print(f"  - Match column: {args.match_column}")
    print(f"  - Label column: label")
    print(f"  - Feature columns: {len(feature_cols)}")
    print(f"\nLabel statistics:")
    print(f"  Range: [{df_combined['label'].min():.4f}, {df_combined['label'].max():.4f}]")
    print(f"  Mean: {df_combined['label'].mean():.4f} ± {df_combined['label'].std():.4f}")
    print(f"\nOutput saved to: {args.output}")
    print("=" * 80)


if __name__ == "__main__":
    main()

