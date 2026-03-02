"""
Feature selection module using mutual information.

Supports both regression and classification tasks with multiple selection criteria.
"""

import pandas as pd
import numpy as np
import json
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
from typing import Dict, List, Optional, Tuple


def select_features_mutual_info(
    X: pd.DataFrame,
    y: pd.Series,
    task: str = 'regression',
    ratio: Optional[float] = None,
    n_features: Optional[int] = None,
    threshold: Optional[float] = None,
    random_state: int = 42
) -> Dict:
    """
    Select features using mutual information.
    
    Args:
        X: Feature matrix (DataFrame)
        y: Target vector (Series)
        task: 'regression' or 'classification'
        ratio: Ratio for selection (e.g., 1.0 means n_features = n_samples)
        n_features: Fixed number of features to select
        threshold: MI threshold (keep features with MI > threshold)
        random_state: Random state for reproducibility
        
    Returns:
        Dictionary with:
            - selected_features: List of selected feature names
            - mi_scores: MI scores for all features
            - selection_info: Information about the selection
    """
    # Validate inputs
    if ratio is None and n_features is None and threshold is None:
        # Default: ratio = 1.0 (samples = features)
        ratio = 1.0
    
    # Handle NaN values
    X_clean = X.fillna(0)
    y_clean = y.fillna(y.median() if task == 'regression' else y.mode()[0] if len(y.mode()) > 0 else 0)
    
    # Calculate mutual information
    print(f"   Calculating mutual information for {len(X.columns)} features...")
    if task == 'regression':
        mi_scores = mutual_info_regression(
            X_clean, y_clean,
            random_state=random_state,
            discrete_features=False
        )
    else:  # classification
        mi_scores = mutual_info_classif(
            X_clean, y_clean,
            random_state=random_state,
            discrete_features=False
        )
    
    # Create DataFrame with feature names and MI scores
    mi_df = pd.DataFrame({
        'feature': X.columns.tolist(),
        'mi_score': mi_scores
    }).sort_values('mi_score', ascending=False)
    
    # Determine number of features to select
    if ratio is not None:
        n_features = int(len(X) / ratio)
    elif n_features is None and threshold is not None:
        n_features = sum(mi_scores > threshold)
    elif n_features is None:
        n_features = len(X)  # Default: keep all
    
    # Ensure n_features doesn't exceed available features
    n_features = min(n_features, len(X.columns))
    
    # Select top features
    selected_features = mi_df.head(n_features)['feature'].tolist()
    
    # Create selection info
    selection_info = {
        'task': task,
        'original_features': len(X.columns),
        'selected_features': len(selected_features),
        'reduction_ratio': len(selected_features) / len(X.columns),
        'selection_method': 'ratio' if ratio is not None else ('fixed' if n_features is not None else 'threshold'),
        'ratio': ratio,
        'n_features': n_features,
        'threshold': threshold,
        'min_mi_score': mi_df.head(n_features)['mi_score'].min(),
        'max_mi_score': mi_df['mi_score'].max(),
        'mean_mi_score': mi_df['mi_score'].mean()
    }
    
    print(f"   Selected {len(selected_features)} features from {len(X.columns)} (ratio: {selection_info['reduction_ratio']:.2%})")
    print(f"   MI score range: [{selection_info['min_mi_score']:.4f}, {selection_info['max_mi_score']:.4f}]")
    
    return {
        'selected_features': selected_features,
        'mi_scores': mi_df,
        'selection_info': selection_info
    }


def save_selected_features(feature_names: List[str], filepath: str, selection_info: Optional[Dict] = None):
    """
    Save selected features to JSON file.
    
    Args:
        feature_names: List of selected feature names
        filepath: Path to save JSON file
        selection_info: Optional selection information to include
    """
    data = {
        'selected_features': feature_names,
        'n_features': len(feature_names)
    }
    
    if selection_info:
        data['selection_info'] = selection_info
    
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"   Selected features saved to {filepath}")


def load_selected_features(filepath: str) -> List[str]:
    """
    Load selected features from JSON file.
    
    Args:
        filepath: Path to JSON file
        
    Returns:
        List of selected feature names
    """
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    selected_features = data['selected_features']
    print(f"   Loaded {len(selected_features)} selected features from {filepath}")
    
    return selected_features

