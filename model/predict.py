"""
Prediction script for trained XGBoost models.

Loads a trained model and makes predictions on new data.
Supports models trained with embeddings and molecular features.
"""

import pandas as pd
import numpy as np
import argparse
import json
import os
import sys
import pickle
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import local modules
from features.featurization import calculate_all_features
from model.xgboost_trainer import load_model
from features.feature_selection import load_selected_features


def load_prediction_data(filepath, smiles_col='smiles'):
    """
    Load prediction data and extract SMILES and embeddings.
    
    Args:
        filepath: Path to CSV file with prediction data
        smiles_col: Name of SMILES column (default: 'smiles')
        
    Returns:
        Tuple of (dataframe, smiles_list, embedding_columns)
    """
    print(f"Loading prediction data from {filepath}...")
    df = pd.read_csv(filepath)
    print(f"  Loaded {len(df)} samples")
    
    # Check for SMILES column
    if smiles_col not in df.columns:
        raise ValueError(f"SMILES column '{smiles_col}' not found in data. Available columns: {list(df.columns[:10])}")
    
    # Find embedding columns
    embedding_cols = [col for col in df.columns if 'embedding' in col.lower()]
    print(f"  Found {len(embedding_cols)} embedding features")
    
    # Extract SMILES
    smiles_list = df[smiles_col].dropna().tolist()
    valid_indices = df.dropna(subset=[smiles_col]).index
    smiles_list = [df.loc[i, smiles_col] for i in valid_indices]
    
    print(f"  Valid SMILES: {len(smiles_list)}")
    
    return df, smiles_list, embedding_cols, valid_indices


def prepare_features_for_prediction(smiles_list, df_data, embedding_cols, valid_indices,
                                   reduced_features_path='reduced_mordred_features.json',
                                   include_map4=True, map4_dimensions=1024):
    """
    Calculate molecular features and combine with embeddings.
    
    Args:
        smiles_list: List of SMILES strings
        df_data: Original dataframe
        embedding_cols: List of embedding column names
        valid_indices: Valid row indices
        reduced_features_path: Path to reduced Mordred features JSON
        include_map4: Whether to include MAP4 fingerprints
        map4_dimensions: MAP4 fingerprint dimensions
        
    Returns:
        Combined feature matrix (DataFrame)
    """
    print("\nCalculating molecular features...")
    
    # Calculate molecular features from SMILES in batches to avoid memory issues
    batch_size = 50  # Process in smaller batches
    molecular_dfs = []
    
    try:
        for i in range(0, len(smiles_list), batch_size):
            batch_smiles = smiles_list[i:i+batch_size]
            batch_num = i//batch_size + 1
            total_batches = (len(smiles_list)-1)//batch_size + 1
            print(f"  Processing batch {batch_num}/{total_batches} ({len(batch_smiles)} molecules)...")
            
            X_batch = calculate_all_features(
                batch_smiles,
                reduced_features_path=reduced_features_path,
                include_map4=include_map4,
                map4_dimensions=map4_dimensions
            )
            molecular_dfs.append(X_batch)
        
        # Concatenate all batches
        X_molecular = pd.concat(molecular_dfs, axis=0, ignore_index=True)
        print(f"  Molecular features: {X_molecular.shape[1]} features")
    except Exception as e:
        print(f"Error calculating molecular features: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    # Extract embeddings for valid samples
    if embedding_cols:
        print(f"\nExtracting embeddings...")
        X_embeddings = df_data.loc[valid_indices, embedding_cols].reset_index(drop=True)
        print(f"  Embedding features: {X_embeddings.shape[1]} features")
        
        # Ensure embeddings are numeric
        for col in X_embeddings.columns:
            X_embeddings[col] = pd.to_numeric(X_embeddings[col], errors='coerce')
        X_embeddings = X_embeddings.fillna(0)
        
        # Combine molecular features and embeddings
        X_combined = pd.concat([X_embeddings, X_molecular], axis=1)
        print(f"  Combined features: {X_combined.shape[1]} total")
    else:
        print("  No embedding columns found, using only molecular features")
        X_combined = X_molecular
    
    # Handle inf and extreme values
    X_combined = X_combined.replace([np.inf, -np.inf], np.nan).fillna(0)
    X_combined = X_combined.clip(lower=-1e10, upper=1e10)
    
    return X_combined


def apply_feature_selection(X, selected_features_path):
    """
    Apply feature selection to match training data.
    
    Args:
        X: Feature matrix
        selected_features_path: Path to selected_features.json
        
    Returns:
        Filtered feature matrix
    """
    if not os.path.exists(selected_features_path):
        print(f"Warning: Selected features file not found: {selected_features_path}")
        print("  Using all available features")
        return X
    
    print(f"\nApplying feature selection...")
    selected_features = load_selected_features(selected_features_path)
    print(f"  Selected features: {len(selected_features)}")
    
    # Check which features are available
    available_features = set(X.columns)
    selected_set = set(selected_features)
    
    missing_features = selected_set - available_features
    if missing_features:
        print(f"  Warning: {len(missing_features)} selected features not found in data")
        print(f"  Missing features (first 10): {list(missing_features)[:10]}")
        # Use only features that are both selected and available
        selected_features = [f for f in selected_features if f in available_features]
        print(f"  Using {len(selected_features)} available selected features")
    
    # Filter to selected features
    X_selected = X[[f for f in selected_features if f in X.columns]]
    
    # Ensure all selected features are present (fill missing with 0)
    for feat in selected_features:
        if feat not in X_selected.columns:
            X_selected[feat] = 0
    
    # Reorder to match training order
    X_selected = X_selected[selected_features]
    
    print(f"  Final feature matrix: {X_selected.shape}")
    
    return X_selected


def make_predictions(model, X, output_path=None):
    """
    Make predictions using the trained model.
    
    Args:
        model: Trained XGBoost model
        X: Feature matrix
        output_path: Optional path to save predictions
        
    Returns:
        Array of predictions
    """
    print("\nMaking predictions...")
    
    # Ensure X is clean
    X = X.fillna(0)
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # Make predictions
    predictions = model.predict(X)
    
    print(f"  Generated {len(predictions)} predictions")
    print(f"  Prediction range: [{predictions.min():.4f}, {predictions.max():.4f}]")
    print(f"  Prediction mean: {predictions.mean():.4f} ± {predictions.std():.4f}")
    
    if output_path:
        pd.DataFrame({'prediction': predictions}).to_csv(output_path, index=False)
        print(f"  Predictions saved to {output_path}")
    
    return predictions


def main():
    """Main prediction pipeline."""
    parser = argparse.ArgumentParser(description='Make predictions using a trained XGBoost model')
    parser.add_argument('--data', type=str, required=True,
                       help='Path to prediction data CSV file')
    parser.add_argument('--model_dir', type=str, required=True,
                       help='Path to directory containing trained model (best_model.pkl)')
    parser.add_argument('--output', type=str, required=True,
                       help='Path to save predictions CSV file')
    parser.add_argument('--smiles_col', type=str, default='smiles',
                       help='Name of SMILES column (default: smiles)')
    parser.add_argument('--reduced_features', type=str, 
                       default='features/reduced_mordred_features.json',
                       help='Path to reduced Mordred features JSON file')
    parser.add_argument('--include_map4', action='store_true', default=True,
                       help='Include MAP4 fingerprints (default: True)')
    parser.add_argument('--map4_dimensions', type=int, default=1024,
                       help='MAP4 fingerprint dimensions (default: 1024)')
    parser.add_argument('--save_features', action='store_true',
                       help='Save the featurized data to CSV')
    parser.add_argument('--calculate_confidence', action='store_true',
                       help='Calculate distance-based confidence scores (requires confidence artifacts from training)')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("XGBoost Model Prediction")
    print("=" * 80)
    print(f"Data: {args.data}")
    print(f"Model directory: {args.model_dir}")
    print(f"Output: {args.output}")
    print("=" * 80)
    
    # Step 1: Load prediction data
    df_data, smiles_list, embedding_cols, valid_indices = load_prediction_data(
        args.data, smiles_col=args.smiles_col
    )
    
    # Step 2: Calculate molecular features and combine with embeddings
    X_combined = prepare_features_for_prediction(
        smiles_list, df_data, embedding_cols, valid_indices,
        reduced_features_path=args.reduced_features,
        include_map4=args.include_map4,
        map4_dimensions=args.map4_dimensions
    )
    
    # Step 3: Apply feature selection (if model was trained with it)
    selected_features_path = os.path.join(args.model_dir, 'selected_features.json')
    if os.path.exists(selected_features_path):
        X_final = apply_feature_selection(X_combined, selected_features_path)
    else:
        print("\nNo feature selection file found, using all features")
        X_final = X_combined
    
    # Step 4: Load trained model
    print(f"\nLoading model from {args.model_dir}...")
    model_path = os.path.join(args.model_dir, 'best_model.pkl')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    model = load_model(model_path)
    
    # Load training metadata if available
    training_results_path = os.path.join(args.model_dir, 'training_results.json')
    task = 'regression'  # default
    if os.path.exists(training_results_path):
        with open(training_results_path, 'r') as f:
            training_info = json.load(f)
        task = training_info.get('task', 'regression')
        print(f"  Model task: {task}")
        print(f"  Training R²: {training_info.get('cv_metrics', {}).get('r2_mean', 'N/A')}")
    
    # Step 5: Make predictions
    predictions = make_predictions(model, X_final)
    
    # Step 5.5: Calculate confidence scores (if enabled and artifacts exist)
    confidence_scores = None
    confidence_levels = None
    if args.calculate_confidence:
        print("\nCalculating confidence scores...")
        from features.confidence_distance import load_confidence_artifacts, predict_confidence
        
        try:
            scaler, X_train_scaled, thresholds, k = load_confidence_artifacts(args.model_dir)
            # Use features from X_final (which already has RDKit columns)
            distances, confidence = predict_confidence(
                X_final, scaler, X_train_scaled, thresholds, k=k
            )
            confidence_scores = distances
            confidence_levels = confidence
            
            conf_counts = pd.Series(confidence).value_counts()
            print(f"  Confidence distribution:")
            for level in ['high', 'medium', 'low']:
                if level in conf_counts:
                    print(f"    {level}: {conf_counts[level]} ({conf_counts[level]/len(confidence)*100:.1f}%)")
        except FileNotFoundError as e:
            print(f"  Warning: Confidence artifacts not found. Skipping confidence calculation.")
            print(f"  {e}")
        except Exception as e:
            print(f"  Warning: Error calculating confidence: {e}")
            print(f"  Skipping confidence calculation.")
    
    # Step 6: Save results with metadata
    print(f"\nSaving predictions to {args.output}...")
    
    # Create output dataframe with original metadata and predictions
    output_df = df_data.loc[valid_indices].copy()
    output_df['prediction'] = predictions
    
    # Add confidence scores if calculated
    if confidence_levels is not None:
        output_df['confidence_level'] = confidence_levels
        output_df['confidence_distance'] = confidence_scores
    
    # Reorder columns to put prediction and confidence near the end
    priority_cols = ['prediction', 'confidence_level', 'confidence_distance']
    other_cols = [c for c in output_df.columns if c not in priority_cols]
    final_cols = other_cols + [c for c in priority_cols if c in output_df.columns]
    output_df = output_df[final_cols]
    
    output_df.to_csv(args.output, index=False)
    print(f"  Saved {len(output_df)} predictions with metadata")
    
    # Save features if requested
    if args.save_features:
        features_output = args.output.replace('.csv', '_features.csv')
        X_final.to_csv(features_output, index=False)
        print(f"  Features saved to {features_output}")
    
    # Print summary
    print("\n" + "=" * 80)
    print("Prediction Complete!")
    print("=" * 80)
    print(f"Predictions saved to: {args.output}")
    print(f"\nPrediction Statistics:")
    print(f"  Count: {len(predictions)}")
    print(f"  Mean: {predictions.mean():.4f}")
    print(f"  Std: {predictions.std():.4f}")
    print(f"  Min: {predictions.min():.4f}")
    print(f"  Max: {predictions.max():.4f}")
    print("=" * 80)


if __name__ == "__main__":
    main()

