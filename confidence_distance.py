"""
Distance-based confidence calculation module.

Calculates confidence scores for predictions based on k-nearest neighbor distances
to the training set in RDKit fingerprint space.
"""

import numpy as np
import pandas as pd
import pickle
import json
import os
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors


def extract_rdkit_features(X):
    """
    Extract RDKit fingerprint columns from feature matrix.
    
    Args:
        X: Feature matrix (DataFrame) containing rdkit_0 to rdkit_2047 columns
        
    Returns:
        DataFrame with only RDKit fingerprint columns (2048 features)
    """
    rdkit_cols = [col for col in X.columns if col.startswith('rdkit_')]
    
    if len(rdkit_cols) == 0:
        raise ValueError("No RDKit fingerprint columns found in feature matrix. Expected columns like rdkit_0, rdkit_1, etc.")
    
    X_rdkit = X[rdkit_cols].copy()
    
    # Ensure numeric types
    for col in X_rdkit.columns:
        X_rdkit[col] = pd.to_numeric(X_rdkit[col], errors='coerce')
    
    # Fill any NaN values
    X_rdkit = X_rdkit.fillna(0)
    
    return X_rdkit


def calculate_knn_distances(X, X_ref, k=5, metric='euclidean'):
    """
    Calculate mean k-nearest neighbor distances.
    
    For each point in X, finds the k nearest neighbors in X_ref and returns
    the mean distance to those neighbors.
    
    Args:
        X: Query points (n_samples, n_features) - numpy array or DataFrame
        X_ref: Reference points (n_ref_samples, n_features) - numpy array or DataFrame
        k: Number of nearest neighbors (default: 5)
        metric: Distance metric (default: 'euclidean')
        
    Returns:
        Array of mean k-NN distances, shape (n_samples,)
    """
    # Convert to numpy if needed
    if isinstance(X, pd.DataFrame):
        X = X.values
    if isinstance(X_ref, pd.DataFrame):
        X_ref = X_ref.values
    
    # Cap k at the number of reference samples
    k_actual = min(k, len(X_ref))
    
    if k_actual < k:
        print(f"   Warning: k={k} requested but only {len(X_ref)} reference samples available. Using k={k_actual}")
    
    # Fit NearestNeighbors on reference set
    nn = NearestNeighbors(n_neighbors=k_actual, metric=metric, algorithm='auto')
    nn.fit(X_ref)
    
    # Find k nearest neighbors for each query point
    distances, indices = nn.kneighbors(X)
    
    # Return mean distance to k nearest neighbors for each point
    mean_distances = distances.mean(axis=1)
    
    return mean_distances


def compute_confidence_thresholds(distances, low_percentile=33.0, high_percentile=67.0):
    """
    Compute percentile thresholds from distance distribution.
    
    Args:
        distances: Array of distances
        low_percentile: Percentile for low/medium boundary (default: 33)
        high_percentile: Percentile for medium/high boundary (default: 67)
        
    Returns:
        Tuple of (low_threshold, high_threshold)
        - distances <= low_threshold → high confidence
        - low_threshold < distances <= high_threshold → medium confidence
        - distances > high_threshold → low confidence
    """
    # Note: lower distance = higher confidence, so we invert the percentile logic
    low_threshold = np.percentile(distances, low_percentile)
    high_threshold = np.percentile(distances, high_percentile)
    
    return low_threshold, high_threshold


def assign_confidence_levels(distances, thresholds):
    """
    Assign confidence levels based on distance thresholds.
    
    Args:
        distances: Array of distances
        thresholds: Tuple of (low_threshold, high_threshold)
        
    Returns:
        Array of confidence level strings ('high', 'medium', 'low')
    """
    low_threshold, high_threshold = thresholds
    
    confidence_levels = np.empty(len(distances), dtype=object)
    confidence_levels[distances <= low_threshold] = 'high'
    confidence_levels[(distances > low_threshold) & (distances <= high_threshold)] = 'medium'
    confidence_levels[distances > high_threshold] = 'low'
    
    return confidence_levels


def fit_confidence_calculator(X_train, k=5, low_percentile=33.0, high_percentile=67.0):
    """
    Fit confidence calculator on training data.
    
    Extracts RDKit features, standardizes them, calculates within-training-set
    k-NN distances, and computes percentile thresholds.
    
    Args:
        X_train: Training feature matrix (DataFrame with rdkit_* columns)
        k: Number of nearest neighbors (default: 5)
        low_percentile: Percentile for low/medium boundary (default: 33)
        high_percentile: Percentile for medium/high boundary (default: 67)
        
    Returns:
        Tuple of (scaler, X_train_scaled, thresholds)
        - scaler: Fitted StandardScaler
        - X_train_scaled: Scaled training RDKit features (numpy array)
        - thresholds: Tuple of (low_threshold, high_threshold)
    """
    print(f"   Fitting confidence calculator on {len(X_train)} training samples...")
    
    # Extract RDKit features
    X_train_rdkit = extract_rdkit_features(X_train)
    print(f"   Extracted {X_train_rdkit.shape[1]} RDKit features")
    
    # Fit scaler on training data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_rdkit)
    
    # Calculate k-NN distances within training set
    # For each training point, find its k+1 nearest neighbors (including itself)
    # and use the mean distance to the k others
    print(f"   Calculating k={k} nearest neighbor distances within training set...")
    k_plus_1 = min(k + 1, len(X_train))  # +1 to exclude self
    
    nn = NearestNeighbors(n_neighbors=k_plus_1, metric='euclidean', algorithm='auto')
    nn.fit(X_train_scaled)
    distances, indices = nn.kneighbors(X_train_scaled)
    
    # Exclude the first neighbor (self, distance=0) and take mean of remaining k
    train_distances = distances[:, 1:].mean(axis=1)
    
    # Compute thresholds from training distance distribution
    thresholds = compute_confidence_thresholds(train_distances, low_percentile, high_percentile)
    print(f"   Distance thresholds: low={thresholds[0]:.4f}, high={thresholds[1]:.4f}")
    
    # Show distribution
    conf_levels = assign_confidence_levels(train_distances, thresholds)
    unique, counts = np.unique(conf_levels, return_counts=True)
    print(f"   Training set confidence distribution (self-consistency check):")
    for level, count in zip(unique, counts):
        print(f"     {level}: {count} ({count/len(train_distances)*100:.1f}%)")
    
    return scaler, X_train_scaled, thresholds


def predict_confidence(X_test, scaler, X_train_scaled, thresholds, k=5):
    """
    Calculate confidence scores for test data.
    
    Args:
        X_test: Test feature matrix (DataFrame with rdkit_* columns)
        scaler: Fitted StandardScaler from training
        X_train_scaled: Scaled training RDKit features (numpy array)
        thresholds: Tuple of (low_threshold, high_threshold)
        k: Number of nearest neighbors (default: 5)
        
    Returns:
        Tuple of (distances, confidence_levels)
        - distances: Array of mean k-NN distances
        - confidence_levels: Array of confidence level strings
    """
    # Extract RDKit features
    X_test_rdkit = extract_rdkit_features(X_test)
    
    # Apply scaler
    X_test_scaled = scaler.transform(X_test_rdkit)
    
    # Calculate k-NN distances to training set
    test_distances = calculate_knn_distances(X_test_scaled, X_train_scaled, k=k, metric='euclidean')
    
    # Assign confidence levels
    confidence_levels = assign_confidence_levels(test_distances, thresholds)
    
    return test_distances, confidence_levels


def save_confidence_artifacts(scaler, X_train_scaled, thresholds, k, output_dir):
    """
    Save confidence calculation artifacts.
    
    Args:
        scaler: Fitted StandardScaler
        X_train_scaled: Scaled training RDKit features
        thresholds: Tuple of (low_threshold, high_threshold)
        k: Number of nearest neighbors
        output_dir: Directory to save artifacts
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save scaler
    scaler_path = os.path.join(output_dir, 'confidence_scaler.pkl')
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    
    # Save training features
    train_features_path = os.path.join(output_dir, 'confidence_train_features.pkl')
    with open(train_features_path, 'wb') as f:
        pickle.dump(X_train_scaled, f)
    
    # Save metadata
    metadata = {
        'k_neighbors': k,
        'low_percentile': None,  # Not stored, only thresholds matter
        'high_percentile': None,
        'thresholds': [float(thresholds[0]), float(thresholds[1])],
        'n_training_samples': len(X_train_scaled),
        'n_features': X_train_scaled.shape[1],
        'feature_type': 'rdkit_atom_pair_2048bit',
        'metric': 'euclidean',
        'normalization': 'standardscaler'
    }
    
    metadata_path = os.path.join(output_dir, 'confidence_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"   Confidence artifacts saved to {output_dir}/")
    print(f"     - confidence_scaler.pkl")
    print(f"     - confidence_train_features.pkl ({X_train_scaled.shape[0]} samples × {X_train_scaled.shape[1]} features)")
    print(f"     - confidence_metadata.json")


def load_confidence_artifacts(model_dir):
    """
    Load confidence calculation artifacts.
    
    Args:
        model_dir: Directory containing saved artifacts
        
    Returns:
        Tuple of (scaler, X_train_scaled, thresholds, k)
    """
    # Load scaler
    scaler_path = os.path.join(model_dir, 'confidence_scaler.pkl')
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Confidence scaler not found: {scaler_path}")
    
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    
    # Load training features
    train_features_path = os.path.join(model_dir, 'confidence_train_features.pkl')
    if not os.path.exists(train_features_path):
        raise FileNotFoundError(f"Training features not found: {train_features_path}")
    
    with open(train_features_path, 'rb') as f:
        X_train_scaled = pickle.load(f)
    
    # Load metadata
    metadata_path = os.path.join(model_dir, 'confidence_metadata.json')
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Confidence metadata not found: {metadata_path}")
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    thresholds = tuple(metadata['thresholds'])
    k = metadata['k_neighbors']
    
    return scaler, X_train_scaled, thresholds, k
