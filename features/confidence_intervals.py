"""
Confidence interval calculation for XGBoost predictions.

Uses quantile regression to estimate prediction intervals.
"""

import xgboost as xgb
import numpy as np
import pandas as pd
from typing import Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


def train_quantile_models(X_train, y_train, X_val, y_val, base_params, 
                         quantiles=[0.05, 0.95], task='regression'):
    """
    Train quantile regression models for confidence intervals.
    
    Args:
        X_train: Training features
        y_train: Training targets
        X_val: Validation features
        y_val: Validation targets
        base_params: Base XGBoost parameters
        quantiles: List of quantiles to predict (default: [0.05, 0.95] for 90% CI)
        task: 'regression' or 'classification'
        
    Returns:
        Dictionary of quantile models: {quantile: model}
    """
    if task != 'regression':
        raise ValueError("Confidence intervals currently only supported for regression")
    
    quantile_models = {}
    
    for quantile in quantiles:
        print(f"   Training quantile model for {quantile:.2f} quantile...")
        
        # Create parameters for quantile regression
        quantile_params = base_params.copy()
        quantile_params['objective'] = f'reg:quantileerror'
        quantile_params['quantile_alpha'] = quantile
        
        # Remove incompatible params
        if 'early_stopping_rounds' in quantile_params:
            early_stopping_rounds = quantile_params.pop('early_stopping_rounds')
        else:
            early_stopping_rounds = 50
        
        # Train quantile model
        model = xgb.XGBRegressor(**quantile_params)
        
        callbacks = []
        if early_stopping_rounds:
            callbacks.append(xgb.callback.EarlyStopping(
                rounds=early_stopping_rounds,
                metric_name='rmse',
                data_name='validation_0',
                save_best=True
            ))
        
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
            callbacks=callbacks
        )
        
        quantile_models[quantile] = model
    
    return quantile_models


def predict_with_confidence(model, quantile_models, X, 
                            confidence_level=0.90):
    """
    Make predictions with confidence intervals.
    
    Args:
        model: Main XGBoost model (for point predictions)
        quantile_models: Dictionary of quantile models
        X: Feature matrix
        confidence_level: Confidence level (default: 0.90 for 90% CI)
        
    Returns:
        Dictionary with:
            - predictions: Point predictions
            - lower_bound: Lower confidence bound
            - upper_bound: Upper confidence bound
            - confidence_interval: Interval width
    """
    X = X.fillna(0)
    
    # Point predictions
    predictions = model.predict(X)
    
    # Quantile predictions
    alpha = 1 - confidence_level
    lower_quantile = alpha / 2
    upper_quantile = 1 - alpha / 2
    
    lower_bound = None
    upper_bound = None
    
    if quantile_models:
        if lower_quantile in quantile_models:
            lower_bound = quantile_models[lower_quantile].predict(X)
        if upper_quantile in quantile_models:
            upper_bound = quantile_models[upper_quantile].predict(X)
    
    # Calculate interval width
    confidence_interval = None
    if lower_bound is not None and upper_bound is not None:
        confidence_interval = upper_bound - lower_bound
    
    return {
        'predictions': predictions,
        'lower_bound': lower_bound,
        'upper_bound': upper_bound,
        'confidence_interval': confidence_interval,
        'confidence_level': confidence_level
    }


def calculate_prediction_intervals_cv(X, y, model_params, cv_folds=5, 
                                     confidence_level=0.90, random_state=42):
    """
    Calculate prediction intervals using cross-validation residuals.
    
    This method uses CV to estimate prediction uncertainty without
    training additional quantile models (faster but less accurate).
    
    Args:
        X: Feature matrix
        y: Target vector
        model_params: XGBoost parameters
        cv_folds: Number of CV folds
        confidence_level: Confidence level (default: 0.90)
        random_state: Random state
        
    Returns:
        Dictionary with prediction intervals
    """
    from sklearn.model_selection import KFold
    
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    
    all_residuals = []
    all_predictions = []
    
    for train_idx, val_idx in kf.split(X):
        X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
        y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
        
        X_train_fold = X_train_fold.fillna(0)
        X_val_fold = X_val_fold.fillna(0)
        
        # Train model
        model = xgb.XGBRegressor(**model_params)
        model.fit(X_train_fold, y_train_fold, verbose=False)
        
        # Predict and calculate residuals
        y_pred_fold = model.predict(X_val_fold)
        residuals = y_val_fold.values - y_pred_fold
        
        all_residuals.extend(residuals)
        all_predictions.extend(y_pred_fold)
    
    # Calculate residual statistics
    all_residuals = np.array(all_residuals)
    residual_std = np.std(all_residuals)
    
    # Calculate z-score for confidence level
    from scipy import stats
    alpha = 1 - confidence_level
    z_score = stats.norm.ppf(1 - alpha/2)
    
    # Prediction intervals based on residual distribution
    interval_width = z_score * residual_std
    
    return {
        'residual_std': residual_std,
        'interval_width': interval_width,
        'z_score': z_score,
        'confidence_level': confidence_level
    }

