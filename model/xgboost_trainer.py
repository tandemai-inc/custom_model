"""
XGBoost model training, evaluation, and persistence module.
"""

import xgboost as xgb
import numpy as np
import pandas as pd
from sklearn.metrics import (
    r2_score, mean_absolute_error, mean_squared_error,
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report
)
from sklearn.model_selection import KFold
from scipy.stats import pearsonr, spearmanr
import pickle
import json
import os


def train_xgboost(X_train, y_train, X_val, y_val, params, 
                 task='regression', early_stopping_rounds=50, verbose=True):
    """
    Train XGBoost model with early stopping.
    
    Args:
        X_train: Training features
        y_train: Training targets
        X_val: Validation features
        y_val: Validation targets
        params: XGBoost hyperparameters
        task: 'regression' or 'classification' (default: 'regression')
        early_stopping_rounds: Early stopping rounds (default: 50)
        verbose: Whether to print training progress (default: True)
        
    Returns:
        Trained model and training history
    """
    # Handle NaN values
    X_train = X_train.fillna(0)
    X_val = X_val.fillna(0)
    
    # Create model
    # Handle early_stopping_rounds if present in params
    model_params = params.copy()
    if 'early_stopping_rounds' in model_params:
        early_stopping_rounds = model_params.pop('early_stopping_rounds')
    else:
        early_stopping_rounds = None
    
    # Create appropriate model type
    if task == 'classification':
        model = xgb.XGBClassifier(**model_params)
    else:
        model = xgb.XGBRegressor(**model_params)
    
    # Train with early stopping
    # Note: In XGBoost 2.0+, early_stopping_rounds is passed in __init__ or via callbacks
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=verbose
    )
    
    # Get training history
    try:
        evals_result = model.evals_result()
        train_rmse = evals_result.get('validation_0', {}).get('rmse', [])
    except:
        train_rmse = []
    
    try:
        best_iter = model.best_iteration
        n_estimators = best_iter + 1 if best_iter is not None else params.get('n_estimators', 100)
    except:
        n_estimators = params.get('n_estimators', 100)
    
    history = {
        'train_rmse': train_rmse,
        'n_estimators': n_estimators
    }
    
    return model, history


def evaluate_model(model, X, y, task='regression'):
    """
    Evaluate model performance.
    
    Args:
        model: Trained XGBoost model
        X: Features
        y: True targets
        task: 'regression' or 'classification' (default: 'regression')
        
    Returns:
        Dictionary with task-appropriate metrics
    """
    X = X.fillna(0)
    y_pred = model.predict(X)
    
    if task == 'classification':
        # Classification metrics
        y_true = y.values if hasattr(y, 'values') else y
        y_pred = y_pred.astype(int)
        
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1': f1_score(y_true, y_pred, average='weighted', zero_division=0),
            'confusion_matrix': confusion_matrix(y_true, y_pred).tolist()
        }
        
        # Add ROC-AUC and PR-AUC for binary classification
        if len(np.unique(y_true)) == 2:
            try:
                y_pred_proba = model.predict_proba(X)[:, 1]
                metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
                metrics['pr_auc'] = average_precision_score(y_true, y_pred_proba)
            except:
                pass
        
        # Add classification report
        try:
            metrics['classification_report'] = classification_report(
                y_true, y_pred, output_dict=True
            )
        except:
            pass
            
    else:
        # Regression metrics
        # Remove any NaN or inf values for correlation calculations
        mask = np.isfinite(y.values) & np.isfinite(y_pred)
        y_clean = y.values[mask]
        y_pred_clean = y_pred[mask]
        
        # Calculate correlations
        pearson_r, pearson_p = pearsonr(y_clean, y_pred_clean) if len(y_clean) > 1 else (np.nan, np.nan)
        spearman_rho, spearman_p = spearmanr(y_clean, y_pred_clean) if len(y_clean) > 1 else (np.nan, np.nan)
        
        metrics = {
            'r2': r2_score(y, y_pred),
            'mae': mean_absolute_error(y, y_pred),
            'rmse': np.sqrt(mean_squared_error(y, y_pred)),
            'pearson_r': pearson_r,
            'pearson_p_value': pearson_p,
            'spearman_rho': spearman_rho,
            'spearman_p_value': spearman_p
        }
    
    return metrics


def cross_validate_model(X, y, params, task='regression', cv_folds=5, random_state=42, output_dir=None):
    """
    Run cross-validation on model.
    
    Args:
        X: Features
        y: Targets
        params: XGBoost hyperparameters
        task: 'regression' or 'classification' (default: 'regression')
        cv_folds: Number of CV folds (default: 5)
        random_state: Random state for reproducibility
        output_dir: Optional directory to save fold predictions
        
    Returns:
        Dictionary with mean and std of CV scores and predictions
    """
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    
    # Store predictions for all folds
    all_predictions = []
    
    if task == 'classification':
        accuracy_scores = []
        precision_scores = []
        recall_scores = []
        f1_scores = []
        roc_auc_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
            X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
            y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
            
            # Train model
            model, _ = train_xgboost(
                X_train_fold, y_train_fold,
                X_val_fold, y_val_fold,
                params,
                task=task,
                verbose=False
            )
            
            # Get predictions
            y_pred = model.predict(X_val_fold)
            y_pred_proba = None
            if hasattr(model, 'predict_proba'):
                y_pred_proba = model.predict_proba(X_val_fold)[:, 1] if task == 'classification' else None
            
            # Store predictions
            fold_predictions = pd.DataFrame({
                'fold': fold,
                'index': val_idx,
                'y_true': y_val_fold.values,
                'y_pred': y_pred
            })
            if y_pred_proba is not None:
                fold_predictions['y_pred_proba'] = y_pred_proba
            all_predictions.append(fold_predictions)
            
            # Save individual fold predictions if output_dir provided
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                fold_path = os.path.join(output_dir, f'cv_fold_{fold}_predictions.csv')
                fold_predictions.to_csv(fold_path, index=False)
            
            # Evaluate
            metrics = evaluate_model(model, X_val_fold, y_val_fold, task=task)
            accuracy_scores.append(metrics['accuracy'])
            precision_scores.append(metrics['precision'])
            recall_scores.append(metrics['recall'])
            f1_scores.append(metrics['f1'])
            if 'roc_auc' in metrics:
                roc_auc_scores.append(metrics['roc_auc'])
            
            print(f"  Fold {fold}/{cv_folds}: Accuracy={metrics['accuracy']:.4f}, "
                  f"Precision={metrics['precision']:.4f}, Recall={metrics['recall']:.4f}, "
                  f"F1={metrics['f1']:.4f}")
        
        results = {
            'accuracy_mean': np.mean(accuracy_scores),
            'accuracy_std': np.std(accuracy_scores),
            'precision_mean': np.mean(precision_scores),
            'precision_std': np.std(precision_scores),
            'recall_mean': np.mean(recall_scores),
            'recall_std': np.std(recall_scores),
            'f1_mean': np.mean(f1_scores),
            'f1_std': np.std(f1_scores),
            'accuracy_scores': accuracy_scores,
            'precision_scores': precision_scores,
            'recall_scores': recall_scores,
            'f1_scores': f1_scores
        }
        
        if roc_auc_scores:
            results['roc_auc_mean'] = np.mean(roc_auc_scores)
            results['roc_auc_std'] = np.std(roc_auc_scores)
            results['roc_auc_scores'] = roc_auc_scores
        
        # Save combined predictions if output_dir provided
        if output_dir and all_predictions:
            combined_predictions = pd.concat(all_predictions, ignore_index=True)
            combined_path = os.path.join(output_dir, 'cv_all_folds_predictions.csv')
            combined_predictions.to_csv(combined_path, index=False)
            print(f"  Combined CV predictions saved to {combined_path}")
        
    else:
        # Regression
        r2_scores = []
        mae_scores = []
        rmse_scores = []
        pearson_scores = []
        spearman_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
            X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
            y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
            
            # Train model
            model, _ = train_xgboost(
                X_train_fold, y_train_fold,
                X_val_fold, y_val_fold,
                params,
                task=task,
                verbose=False
            )
            
            # Get predictions
            y_pred = model.predict(X_val_fold)
            
            # Store predictions
            fold_predictions = pd.DataFrame({
                'fold': fold,
                'index': val_idx,
                'y_true': y_val_fold.values,
                'y_pred': y_pred
            })
            all_predictions.append(fold_predictions)
            
            # Save individual fold predictions if output_dir provided
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                fold_path = os.path.join(output_dir, f'cv_fold_{fold}_predictions.csv')
                fold_predictions.to_csv(fold_path, index=False)
            
            # Evaluate
            metrics = evaluate_model(model, X_val_fold, y_val_fold, task=task)
            r2_scores.append(metrics['r2'])
            mae_scores.append(metrics['mae'])
            rmse_scores.append(metrics['rmse'])
            pearson_scores.append(metrics['pearson_r'])
            spearman_scores.append(metrics['spearman_rho'])
            
            print(f"  Fold {fold}/{cv_folds}: R²={metrics['r2']:.4f}, MAE={metrics['mae']:.4f}, RMSE={metrics['rmse']:.4f}, "
                  f"Pearson r={metrics['pearson_r']:.4f}, Spearman ρ={metrics['spearman_rho']:.4f}")
        
        results = {
            'r2_mean': np.mean(r2_scores),
            'r2_std': np.std(r2_scores),
            'mae_mean': np.mean(mae_scores),
            'mae_std': np.std(mae_scores),
            'rmse_mean': np.mean(rmse_scores),
            'rmse_std': np.std(rmse_scores),
            'pearson_r_mean': np.mean(pearson_scores),
            'pearson_r_std': np.std(pearson_scores),
            'spearman_rho_mean': np.mean(spearman_scores),
            'spearman_rho_std': np.std(spearman_scores),
            'r2_scores': r2_scores,
            'mae_scores': mae_scores,
            'rmse_scores': rmse_scores,
            'pearson_scores': pearson_scores,
            'spearman_scores': spearman_scores
        }
        
        # Save combined predictions if output_dir provided
        if output_dir and all_predictions:
            combined_predictions = pd.concat(all_predictions, ignore_index=True)
            combined_path = os.path.join(output_dir, 'cv_all_folds_predictions.csv')
            combined_predictions.to_csv(combined_path, index=False)
            print(f"  Combined CV predictions saved to {combined_path}")
    
    return results


def save_model(model, filepath):
    """
    Save trained model to file.
    
    Args:
        model: Trained XGBoost model
        filepath: Path to save model
    """
    os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
    
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)
    
    print(f"Model saved to {filepath}")


def load_model(filepath):
    """
    Load saved model from file.
    
    Args:
        filepath: Path to saved model
        
    Returns:
        Loaded XGBoost model
    """
    with open(filepath, 'rb') as f:
        model = pickle.load(f)
    
    print(f"Model loaded from {filepath}")
    return model


def get_feature_importance(model, feature_names=None, importance_type='gain', top_n=50):
    """
    Extract and rank feature importance.
    
    Args:
        model: Trained XGBoost model
        feature_names: List of feature names (optional)
        importance_type: Type of importance ('gain', 'weight', 'cover')
        top_n: Number of top features to return (default: 50)
        
    Returns:
        DataFrame with feature importance rankings
    """
    # Get feature importance from model
    # Returns dict with keys that could be either:
    # - Feature names (if feature names were provided during training): 'mordred_SpAbs_A', etc.
    # - Feature indices (if no names provided): 'f0', 'f1', etc.
    importance = model.get_booster().get_score(importance_type=importance_type)
    
    # Check if keys are feature names or indices
    # If first key starts with 'f' and is followed by digits, it's an index
    first_key = list(importance.keys())[0] if importance else None
    is_indexed = first_key and first_key.startswith('f') and first_key[1:].isdigit()
    
    # Convert to DataFrame
    if feature_names is None:
        if is_indexed:
            # Use feature indices - extract index from keys (e.g., 'f0' -> 0, 'f1' -> 1)
            df_importance = pd.DataFrame({
                'feature': [f'feature_{int(k[1:])}' for k in importance.keys()],
                'importance': list(importance.values())
            })
        else:
            # Keys are already feature names
            df_importance = pd.DataFrame({
                'feature': list(importance.keys()),
                'importance': list(importance.values())
            })
    else:
        if is_indexed:
            # Map from importance dict (keys are 'f0', 'f1', etc.) to feature names
            # Create a mapping: feature index -> importance value
            importance_dict = {int(k[1:]): v for k, v in importance.items()}
            
            # Map each feature by its index
            df_importance = pd.DataFrame({
                'feature': feature_names,
                'importance': [importance_dict.get(i, 0) for i in range(len(feature_names))]
            })
        else:
            # Keys are feature names - map directly
            df_importance = pd.DataFrame({
                'feature': feature_names,
                'importance': [importance.get(fname, 0) for fname in feature_names]
            })
    
    # Sort by importance (descending)
    df_importance = df_importance.sort_values('importance', ascending=False)
    
    # Get top N
    df_importance_top = df_importance.head(top_n)
    
    return df_importance, df_importance_top

