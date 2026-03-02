"""
Optuna hyperparameter optimization module with cross-validation.

Uses cross-validation scores for robust hyperparameter selection.
"""

import optuna
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import (
    r2_score, mean_absolute_error, mean_squared_error,
    accuracy_score, f1_score, roc_auc_score
)
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')


def create_objective(X, y, task='regression', cv_folds=5, metric=None, random_state=42):
    """
    Create Optuna objective function with cross-validation.
    
    Args:
        X: Feature matrix (DataFrame or array)
        y: Target vector (Series or array)
        task: 'regression' or 'classification' (default: 'regression')
        cv_folds: Number of CV folds (default: 5)
        metric: Metric to optimize (default: 'r2' for regression, 'f1' for classification)
        random_state: Random state for reproducibility
        
    Returns:
        Objective function for Optuna
    """
    # Set default metric based on task
    if metric is None:
        metric = 'f1' if task == 'classification' else 'r2'
    
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    
    def objective(trial):
        # Suggest hyperparameters
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 2000),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 7),
            'gamma': trial.suggest_float('gamma', 0, 0.5),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 1),
            'reg_lambda': trial.suggest_float('reg_lambda', 0, 1),
            'random_state': random_state,
            'n_jobs': -1,
            'tree_method': 'hist'  # Faster training
        }
        
        # Cross-validation scores
        cv_scores = []
        
        for train_idx, val_idx in kf.split(X):
            X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
            y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
            
            # Handle NaN, inf, and extreme values
            X_train_fold = X_train_fold.replace([np.inf, -np.inf], np.nan)
            X_val_fold = X_val_fold.replace([np.inf, -np.inf], np.nan)
            X_train_fold = X_train_fold.fillna(0)
            X_val_fold = X_val_fold.fillna(0)
            # Clip extreme values
            X_train_fold = X_train_fold.clip(lower=-1e10, upper=1e10)
            X_val_fold = X_val_fold.clip(lower=-1e10, upper=1e10)
            
            # Create appropriate model type
            if task == 'classification':
                model = xgb.XGBClassifier(**params)
            else:
                model = xgb.XGBRegressor(**params)
            
            model.fit(
                X_train_fold, y_train_fold,
                eval_set=[(X_val_fold, y_val_fold)],
                verbose=False
            )
            
            # Predict and calculate metric
            y_pred = model.predict(X_val_fold)
            
            if task == 'classification':
                if metric == 'f1':
                    score = f1_score(y_val_fold, y_pred, average='weighted', zero_division=0)
                elif metric == 'accuracy':
                    score = accuracy_score(y_val_fold, y_pred)
                elif metric == 'roc_auc':
                    if len(np.unique(y_val_fold)) == 2:
                        y_pred_proba = model.predict_proba(X_val_fold)[:, 1]
                        score = roc_auc_score(y_val_fold, y_pred_proba)
                    else:
                        score = 0.0
                else:
                    raise ValueError(f"Unknown classification metric: {metric}")
            else:
                # Regression
                if metric == 'r2':
                    score = r2_score(y_val_fold, y_pred)
                elif metric == 'mae':
                    score = -mean_absolute_error(y_val_fold, y_pred)  # Negative for minimization
                elif metric == 'rmse':
                    score = -np.sqrt(mean_squared_error(y_val_fold, y_pred))  # Negative for minimization
                else:
                    raise ValueError(f"Unknown regression metric: {metric}")
            
            cv_scores.append(score)
        
        # Return mean CV score
        return np.mean(cv_scores)
    
    return objective


def optimize_hyperparameters(X, y, task='regression', n_trials=100, cv_folds=5, metric=None, 
                             direction='maximize', random_state=42, 
                             study_name='xgboost_optuna'):
    """
    Optimize XGBoost hyperparameters using Optuna with cross-validation.
    
    Args:
        X: Feature matrix (DataFrame or array)
        y: Target vector (Series or array)
        task: 'regression' or 'classification' (default: 'regression')
        n_trials: Number of Optuna trials (default: 100)
        cv_folds: Number of CV folds (default: 5)
        metric: Metric to optimize (default: 'r2' for regression, 'f1' for classification)
        direction: Optimization direction ('maximize' or 'minimize')
        random_state: Random state for reproducibility
        study_name: Name for Optuna study
        
    Returns:
        Dictionary with best hyperparameters and study object
    """
    # Set default metric based on task
    if metric is None:
        metric = 'f1' if task == 'classification' else 'r2'
    
    print("=" * 80)
    print("Optuna Hyperparameter Optimization")
    print("=" * 80)
    print(f"  Task: {task}")
    print(f"  Metric: {metric}")
    print(f"  CV folds: {cv_folds}")
    print(f"  Trials: {n_trials}")
    
    # Determine direction based on metric
    if task == 'classification':
        # All classification metrics are maximized
        direction = 'maximize'
    else:
        # Regression
        if metric == 'r2':
            direction = 'maximize'
        else:  # mae, rmse
            direction = 'minimize'
    
    print(f"  Direction: {direction}")
    
    # Create study
    study = optuna.create_study(
        direction=direction,
        study_name=study_name,
        sampler=optuna.samplers.TPESampler(seed=random_state)
    )
    
    # Create objective function
    objective = create_objective(X, y, task=task, cv_folds=cv_folds, metric=metric, 
                                random_state=random_state)
    
    # Optimize
    print("\nStarting optimization...")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    # Get best parameters
    best_params = study.best_params
    best_value = study.best_value
    
    print("\n" + "=" * 80)
    print("Optimization Complete")
    print("=" * 80)
    print(f"Best {metric}: {best_value:.6f}")
    print(f"\nBest hyperparameters:")
    for key, value in best_params.items():
        print(f"  {key}: {value}")
    
    return {
        'best_params': best_params,
        'best_value': best_value,
        'study': study
    }

