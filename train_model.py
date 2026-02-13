"""
Main training script for XGBoost model with Optuna hyperparameter optimization.

Combines all molecular features and trains XGBoost model with cross-validation.
"""

import pandas as pd
import numpy as np
import argparse
import json
import os
import sys
import pickle
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class Tee:
    """Write to multiple file-like objects (e.g. stdout and a log file)."""
    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()

    def flush(self):
        for f in self.files:
            f.flush()


class TeeStdout:
    """Context manager: always write a copy of stdout to output_dir/training.log."""
    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.log_file = None
        self.original_stdout = None

    def __enter__(self):
        self.original_stdout = sys.stdout
        self.log_path = os.path.join(self.output_dir, 'training.log')
        self.log_file = open(self.log_path, 'w', encoding='utf-8')
        sys.stdout = Tee(self.original_stdout, self.log_file)
        return self

    def __exit__(self, *args):
        sys.stdout = self.original_stdout
        if self.log_file:
            self.log_file.close()
        return False


# Import local modules
from featurization import calculate_all_features
from optuna_tuning import optimize_hyperparameters
from xgboost_trainer import (
    train_xgboost, evaluate_model, cross_validate_model,
    save_model, get_feature_importance
)
from feature_selection import (
    select_features_mutual_info, save_selected_features, load_selected_features
)
try:
    from confidence_intervals import (
        train_quantile_models, predict_with_confidence, calculate_prediction_intervals_cv
    )
    CONFIDENCE_AVAILABLE = True
except ImportError:
    CONFIDENCE_AVAILABLE = False
    print("Warning: confidence_intervals module not available")
try:
    from confidence_intervals import (
        train_quantile_models, predict_with_confidence, calculate_prediction_intervals_cv
    )
    CONFIDENCE_AVAILABLE = True
except ImportError:
    CONFIDENCE_AVAILABLE = False
    print("Warning: confidence_intervals module not available")


def main():
    """Main training pipeline."""
    parser = argparse.ArgumentParser(description='Train XGBoost model with Optuna optimization')
    parser.add_argument('--data', type=str, default='train_val_features.csv',
                       help='Path to training data CSV file')
    parser.add_argument('--n_trials', type=int, default=100,
                       help='Number of Optuna trials (default: 100)')
    parser.add_argument('--cv_folds', type=int, default=5,
                       help='Number of CV folds (default: 5)')
    parser.add_argument('--task', type=str, default='regression', choices=['regression', 'classification'],
                       help='Task type: regression or classification (default: regression)')
    parser.add_argument('--metric', type=str, default='r2', 
                       help='Metric to optimize (default: r2 for regression, f1 for classification)')
    parser.add_argument('--classification_threshold', type=float, default=None,
                       help='Custom threshold for classification (default: median split)')
    parser.add_argument('--use_feature_selection', action='store_true',
                       help='Enable mutual information feature selection')
    parser.add_argument('--feature_selection_ratio', type=float, default=1.0,
                       help='Feature selection ratio (default: 1.0 means n_features = n_samples)')
    parser.add_argument('--feature_selection_n', type=int, default=None,
                       help='Fixed number of features to select (alternative to ratio)')
    parser.add_argument('--feature_selection_threshold', type=float, default=None,
                       help='MI threshold for feature selection (alternative to ratio/n)')
    parser.add_argument('--saved_features', type=str, default=None,
                       help='Path to previously saved feature selection JSON file')
    parser.add_argument('--output_dir', type=str, default='results',
                       help='Output directory for results (default: results)')
    parser.add_argument('--reduced_features', type=str, 
                       default='reduced_mordred_features.json',
                       help='Path to reduced Mordred features JSON file')
    parser.add_argument('--include_map4', action='store_true', default=True,
                       help='Include MAP4 fingerprints (default: True)')
    parser.add_argument('--map4_dimensions', type=int, default=1024,
                       help='MAP4 fingerprint dimensions (default: 1024)')
    parser.add_argument('--test_size', type=float, default=0.2,
                       help='Test set size (default: 0.2)')
    parser.add_argument('--random_state', type=int, default=42,
                       help='Random state for reproducibility (default: 42)')
    parser.add_argument('--skip_optuna', action='store_true',
                       help='Skip Optuna optimization and use default hyperparameters')
    parser.add_argument('--confidence_intervals', action='store_true',
                       help='Calculate confidence intervals for predictions (default: False, can be slow)')
    parser.add_argument('--confidence_level', type=float, default=0.90,
                       help='Confidence level for intervals (default: 0.90 for 90%% CI)')
    parser.add_argument('--cache_features', action='store_true', default=True,
                       help='Cache featurized data in results folder (default: True)')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    with TeeStdout(args.output_dir):
        print("=" * 80)
        print("XGBoost Training Pipeline")
        print("=" * 80)
        print(f"Data: {args.data}")
        print(f"Task: {args.task}")
        print(f"Optuna trials: {args.n_trials}")
        print(f"CV folds: {args.cv_folds}")
        print(f"Metric: {args.metric}")
        print(f"Feature selection: {args.use_feature_selection}")
        if args.use_feature_selection:
            if args.saved_features:
                print(f"  Using saved features: {args.saved_features}")
            else:
                print(f"  Selection ratio: {args.feature_selection_ratio}")
        print(f"Output directory: {args.output_dir}")
        print("=" * 80)
        
        # Step 1: Load data
        print("\n1. Loading data...")
        try:
            df = pd.read_csv(args.data)
            print(f"   Loaded {len(df)} samples")
        except FileNotFoundError:
            print(f"Error: Data file not found: {args.data}")
            return
        
        # Extract SMILES and labels
        if 'smiles' not in df.columns or 'label' not in df.columns:
            print("Error: Data must contain 'smiles' and 'label' columns")
            return
        
        smiles_list = df['smiles'].dropna().tolist()
        y = df['label'].dropna()
        
        # Align indices
        valid_indices = df.dropna(subset=['smiles', 'label']).index
        smiles_list = [df.loc[i, 'smiles'] for i in valid_indices]
        y = df.loc[valid_indices, 'label']
        
        print(f"   Valid samples: {len(smiles_list)}")
        print(f"   Target range: [{y.min():.2f}, {y.max():.2f}]")
        print(f"   Target mean: {y.mean():.2f} ± {y.std():.2f}")
    
        # Step 1.5: Detect/Set task type and apply classification threshold if needed
        task = args.task
        classification_threshold = None
    
        if task == 'classification':
            # Apply classification threshold (median split by default)
            if args.classification_threshold is not None:
                classification_threshold = args.classification_threshold
            else:
                classification_threshold = y.median()
        
            print(f"\n1.5. Applying classification threshold...")
            print(f"   Threshold: {classification_threshold:.4f} (median split)")
            y = (y >= classification_threshold).astype(int)
        
            class_counts = y.value_counts().sort_index()
            print(f"   Class distribution:")
            for class_val, count in class_counts.items():
                print(f"     Class {class_val}: {count} samples ({count/len(y)*100:.1f}%)")
    
        # Step 2: Load or calculate features
        print("\n2. Loading/calculating features...")
    
        # Check if CSV already contains feature columns (e.g., from concatenate_peptide_features.py)
        metadata_cols = ['smiles', 'label']
        if 'fold' in df.columns:
            metadata_cols.append('fold')
    
        # Also exclude common label-like column names to prevent data leakage
        label_like_patterns = ['permeability', 'pampa', 'caco2', 'mdck', 'rrck']
        for col in df.columns:
            col_lower = col.lower()
            if any(pattern in col_lower for pattern in label_like_patterns) and col != 'label':
                if col not in metadata_cols:
                    metadata_cols.append(col)
                    print(f"   Warning: Excluding potential label column from features: {col}")
    
        existing_feature_cols = [col for col in df.columns if col not in metadata_cols]
    
        if len(existing_feature_cols) > 0:
            # Use pre-computed features from CSV
            print(f"   Found {len(existing_feature_cols)} pre-computed features in CSV")
        
            # Check for embedding features
            embedding_cols = [col for col in existing_feature_cols if 'embedding' in col.lower()]
            molecular_cols = [col for col in existing_feature_cols if 'embedding' not in col.lower()]
        
            if embedding_cols:
                print(f"   - Embedding features: {len(embedding_cols)}")
            if molecular_cols:
                print(f"   - Molecular features: {len(molecular_cols)}")
        
            # Extract features for valid samples only
            X = df.loc[valid_indices, existing_feature_cols].reset_index(drop=True)
        
            # Ensure numeric types and handle inf/extreme values
            for col in X.columns:
                X[col] = pd.to_numeric(X[col], errors='coerce')
        
            # Replace inf and -inf with NaN, then fill
            X = X.replace([np.inf, -np.inf], np.nan)
            X = X.fillna(0)
        
            # Clip extreme values to prevent overflow
            # Clip to reasonable range (e.g., -1e10 to 1e10)
            X = X.clip(lower=-1e10, upper=1e10)
        
            print(f"   Using {X.shape[1]} pre-computed features from CSV")
        else:
            # No pre-computed features, calculate from SMILES
            print("   No pre-computed features found, calculating from SMILES...")
            try:
                X = calculate_all_features(
                    smiles_list,
                    reduced_features_path=args.reduced_features,
                    include_map4=args.include_map4,
                    map4_dimensions=args.map4_dimensions
                )
                print(f"   Features calculated: {X.shape[1]}")
            except Exception as e:
                print(f"Error calculating features: {e}")
                import traceback
                traceback.print_exc()
                return
    
        # Step 2.5: Feature selection
        if args.use_feature_selection:
            print("\n2.5. Feature selection...")
            try:
                if args.saved_features:
                    # Load previously selected features
                    selected_feature_names = load_selected_features(args.saved_features)
                    X = X[selected_feature_names]
                    print(f"   Using {len(selected_feature_names)} pre-selected features")
                else:
                    # Perform feature selection
                    selection_result = select_features_mutual_info(
                        X, y,
                        task=task,
                        ratio=args.feature_selection_ratio if args.feature_selection_n is None else None,
                        n_features=args.feature_selection_n,
                        threshold=args.feature_selection_threshold,
                        random_state=args.random_state
                    )
                
                    selected_feature_names = selection_result['selected_features']
                    X = X[selected_feature_names]
                
                    # Save selected features
                    features_path = os.path.join(args.output_dir, 'selected_features.json')
                    save_selected_features(
                        selected_feature_names,
                        features_path,
                        selection_result['selection_info']
                    )
                
                    # Save MI scores
                    mi_scores_path = os.path.join(args.output_dir, 'feature_selection_scores.csv')
                    selection_result['mi_scores'].to_csv(mi_scores_path, index=False)
                    print(f"   MI scores saved to {mi_scores_path}")
                
            except Exception as e:
                print(f"Error in feature selection: {e}")
                import traceback
                traceback.print_exc()
                return
    
        # Step 2.6: Cache featurized data (if enabled)
        if args.cache_features:
            print("\n2.6. Caching featurized data...")
            try:
                cache_dir = os.path.join(args.output_dir, 'feature_cache')
                os.makedirs(cache_dir, exist_ok=True)
            
                # Save feature matrix and labels
                X.to_csv(os.path.join(cache_dir, 'X_features.csv'), index=False)
                y.to_csv(os.path.join(cache_dir, 'y_labels.csv'), index=False)
            
                # Save feature names
                feature_names_path = os.path.join(cache_dir, 'feature_names.json')
                with open(feature_names_path, 'w') as f:
                    json.dump({'feature_names': X.columns.tolist()}, f, indent=2)
            
                # Save metadata
                cache_metadata = {
                    'n_samples': len(X),
                    'n_features': X.shape[1],
                    'feature_names': X.columns.tolist(),
                    'task': task
                }
                with open(os.path.join(cache_dir, 'cache_metadata.json'), 'w') as f:
                    json.dump(cache_metadata, f, indent=2)
            
                print(f"   Featurized data cached to {cache_dir}/")
                print(f"   - X_features.csv: {X.shape[0]} samples × {X.shape[1]} features")
                print(f"   - y_labels.csv: {len(y)} labels")
                print(f"   - feature_names.json: Feature names")
            except Exception as e:
                print(f"   Warning: Could not cache features: {e}")
    
        # Step 3: Split data
        print("\n3. Splitting data...")
        from sklearn.model_selection import train_test_split
    
        if args.test_size == 0:
            # Use all data for training (no test set)
            X_train, X_test, y_train, y_test = X, None, y, None
            print(f"   Train set: {len(X_train)} samples (all data)")
            print(f"   Test set: None (test_size=0)")
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=args.test_size, random_state=args.random_state,
                stratify=y if task == 'classification' else None
            )
            print(f"   Train set: {len(X_train)} samples")
            print(f"   Test set: {len(X_test)} samples")
    
        # Step 4: Optuna optimization (or use defaults)
        if args.skip_optuna:
            print("\n4. Skipping Optuna optimization (using default hyperparameters)...")
            # Default hyperparameters
            best_params = {
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'min_child_weight': 1,
                'gamma': 0,
                'reg_alpha': 0,
                'reg_lambda': 1,
                'random_state': args.random_state
            }
            best_value = None
            print("   Using default hyperparameters")
        else:
            print("\n4. Optuna hyperparameter optimization...")
            try:
                # Set default metric based on task if not specified
                metric = args.metric
                if metric == 'r2' and task == 'classification':
                    metric = 'f1'
                elif metric not in ['r2', 'mae', 'rmse'] and task == 'regression':
                    metric = 'r2'
            
                optuna_results = optimize_hyperparameters(
                    X_train, y_train,
                    task=task,
                    n_trials=args.n_trials,
                    cv_folds=args.cv_folds,
                    metric=metric,
                    random_state=args.random_state
                )
            
                best_params = optuna_results['best_params']
                best_value = optuna_results['best_value']
            
            except Exception as e:
                print(f"Error in Optuna optimization: {e}")
                import traceback
                traceback.print_exc()
                return
    
        # Save hyperparameters
        os.makedirs(args.output_dir, exist_ok=True)  # Ensure directory exists
        hyperparams_path = os.path.join(args.output_dir, 'best_hyperparameters.json')
        with open(hyperparams_path, 'w') as f:
            json.dump(best_params, f, indent=2)
        print(f"   Hyperparameters saved to {hyperparams_path}")
    
        # Step 5: Train final model
        print("\n5. Training final model...")
        try:
            # Split train into train/val for early stopping
            X_train_final, X_val_final, y_train_final, y_val_final = train_test_split(
                X_train, y_train, test_size=0.2, random_state=args.random_state
            )
        
            # Add early stopping rounds to params
            final_params = best_params.copy()
            final_params['early_stopping_rounds'] = 50
        
            model, history = train_xgboost(
                X_train_final, y_train_final,
                X_val_final, y_val_final,
                final_params,
                task=task,
                verbose=True
            )
        
            print(f"   Model trained with {history['n_estimators']} estimators")
        
            # Step 5.5: Train quantile models for confidence intervals (if enabled)
            quantile_models = None
            if args.confidence_intervals and task == 'regression' and CONFIDENCE_AVAILABLE:
                print("\n5.5. Training quantile models for confidence intervals...")
                try:
                    alpha = 1 - args.confidence_level
                    quantiles = [alpha / 2, 1 - alpha / 2]  # e.g., [0.05, 0.95] for 90% CI
                
                    quantile_models = train_quantile_models(
                        X_train_final, y_train_final,
                        X_val_final, y_val_final,
                        final_params,
                        quantiles=quantiles,
                        task=task
                    )
                
                    # Save quantile models
                    quantile_models_path = os.path.join(args.output_dir, 'quantile_models.pkl')
                    with open(quantile_models_path, 'wb') as f:
                        pickle.dump(quantile_models, f)
                    print(f"   Quantile models saved to {quantile_models_path}")
                
                except Exception as e:
                    print(f"   Warning: Could not train quantile models: {e}")
                    print(f"   Continuing without confidence intervals...")
                    quantile_models = None
            elif args.confidence_intervals and task != 'regression':
                print("\n5.5. Skipping confidence intervals (only supported for regression)")
            elif args.confidence_intervals and not CONFIDENCE_AVAILABLE:
                print("\n5.5. Skipping confidence intervals (module not available)")
        
        except Exception as e:
            print(f"Error training model: {e}")
            import traceback
            traceback.print_exc()
            return
    
        # Step 6: Evaluate on test set (if available)
        test_predictions_with_ci = None
        if X_test is not None and y_test is not None:
            print("\n6. Evaluating on test set...")
            test_metrics = evaluate_model(model, X_test, y_test, task=task)
        
            # Calculate confidence intervals for test predictions (if enabled)
            if args.confidence_intervals and task == 'regression' and quantile_models is not None:
                print("   Calculating confidence intervals for test predictions...")
                try:
                    ci_results = predict_with_confidence(
                        model, quantile_models, X_test,
                        confidence_level=args.confidence_level
                    )
                
                    # Create predictions DataFrame with confidence intervals
                    test_predictions_with_ci = pd.DataFrame({
                        'y_true': y_test.values,
                        'y_pred': ci_results['predictions'],
                        'lower_bound': ci_results['lower_bound'],
                        'upper_bound': ci_results['upper_bound'],
                        'confidence_interval': ci_results['confidence_interval'],
                        'confidence_level': args.confidence_level
                    })
                
                    # Save test predictions with confidence intervals
                    test_pred_path = os.path.join(args.output_dir, 'test_predictions_with_ci.csv')
                    test_predictions_with_ci.to_csv(test_pred_path, index=False)
                    print(f"   Test predictions with confidence intervals saved to {test_pred_path}")
                
                    # Add CI statistics to test_metrics
                    test_metrics['confidence_intervals'] = {
                        'mean_interval_width': float(np.mean(ci_results['confidence_interval'])),
                        'median_interval_width': float(np.median(ci_results['confidence_interval'])),
                        'confidence_level': args.confidence_level
                    }
                
                except Exception as e:
                    print(f"   Warning: Could not calculate confidence intervals: {e}")
        
            if task == 'regression':
                print(f"   Test R²: {test_metrics['r2']:.4f}")
                print(f"   Test MAE: {test_metrics['mae']:.4f}")
                print(f"   Test RMSE: {test_metrics['rmse']:.4f}")
                print(f"   Test Pearson r: {test_metrics['pearson_r']:.4f} (p={test_metrics['pearson_p_value']:.2e})")
                print(f"   Test Spearman ρ: {test_metrics['spearman_rho']:.4f} (p={test_metrics['spearman_p_value']:.2e})")
                if 'confidence_intervals' in test_metrics:
                    ci_info = test_metrics['confidence_intervals']
                    print(f"   Mean CI width: {ci_info['mean_interval_width']:.4f}")
            else:  # classification
                print(f"   Test Accuracy: {test_metrics['accuracy']:.4f}")
                print(f"   Test Precision: {test_metrics['precision']:.4f}")
                print(f"   Test Recall: {test_metrics['recall']:.4f}")
                print(f"   Test F1: {test_metrics['f1']:.4f}")
                if 'roc_auc' in test_metrics:
                    print(f"   Test ROC-AUC: {test_metrics['roc_auc']:.4f}")
                if 'pr_auc' in test_metrics:
                    print(f"   Test PR-AUC: {test_metrics['pr_auc']:.4f}")
        else:
            print("\n6. Skipping test set evaluation (test_size=0, using all data for training)")
            test_metrics = None
    
        # Step 7: Cross-validation evaluation
        print("\n7. Cross-validation evaluation...")
        cv_results = cross_validate_model(
            X_train, y_train,
            best_params,
            task=task,
            cv_folds=args.cv_folds,
            random_state=args.random_state,
            output_dir=args.output_dir
        )
    
        print(f"\n   CV Results:")
        if task == 'regression':
            print(f"     R²: {cv_results['r2_mean']:.4f} ± {cv_results['r2_std']:.4f}")
            print(f"     MAE: {cv_results['mae_mean']:.4f} ± {cv_results['mae_std']:.4f}")
            print(f"     RMSE: {cv_results['rmse_mean']:.4f} ± {cv_results['rmse_std']:.4f}")
            print(f"     Pearson r: {cv_results['pearson_r_mean']:.4f} ± {cv_results['pearson_r_std']:.4f}")
            print(f"     Spearman ρ: {cv_results['spearman_rho_mean']:.4f} ± {cv_results['spearman_rho_std']:.4f}")
        else:  # classification
            print(f"     Accuracy: {cv_results['accuracy_mean']:.4f} ± {cv_results['accuracy_std']:.4f}")
            print(f"     Precision: {cv_results['precision_mean']:.4f} ± {cv_results['precision_std']:.4f}")
            print(f"     Recall: {cv_results['recall_mean']:.4f} ± {cv_results['recall_std']:.4f}")
            print(f"     F1: {cv_results['f1_mean']:.4f} ± {cv_results['f1_std']:.4f}")
            if 'roc_auc_mean' in cv_results:
                print(f"     ROC-AUC: {cv_results['roc_auc_mean']:.4f} ± {cv_results['roc_auc_std']:.4f}")
    
        # Step 8: Feature importance
        print("\n8. Extracting feature importance...")
        feature_names = X.columns.tolist()
    
        # Save confusion matrix for classification (if test set exists)
        if task == 'classification' and test_metrics and 'confusion_matrix' in test_metrics:
            cm_path = os.path.join(args.output_dir, 'confusion_matrix.csv')
            cm_df = pd.DataFrame(
                test_metrics['confusion_matrix'],
                index=[f'True_{i}' for i in range(len(test_metrics['confusion_matrix']))],
                columns=[f'Pred_{i}' for i in range(len(test_metrics['confusion_matrix'][0]))]
            )
            cm_df.to_csv(cm_path)
            print(f"   Confusion matrix saved to {cm_path}")
        df_importance, df_importance_top = get_feature_importance(
            model, feature_names, top_n=50
        )
    
        importance_path = os.path.join(args.output_dir, 'feature_importance.csv')
        df_importance.to_csv(importance_path, index=False)
        print(f"   Feature importance saved to {importance_path}")
    
        print(f"\n   Top 10 features:")
        for idx, row in df_importance_top.head(10).iterrows():
            print(f"     {row['feature']}: {row['importance']:.4f}")
    
        # Step 9: Save model
        print("\n9. Saving model...")
        model_path = os.path.join(args.output_dir, 'best_model.pkl')
        save_model(model, model_path)
    
        # Step 10: Save results
        print("\n10. Saving results...")
        results = {
            'timestamp': datetime.now().isoformat(),
            'data_file': args.data,
            'task': task,
            'classification_threshold': classification_threshold if task == 'classification' else None,
            'n_samples': len(smiles_list),
            'n_features': X.shape[1],
            'feature_selection_used': args.use_feature_selection,
            'test_size': args.test_size,
            'n_trials': args.n_trials,
            'cv_folds': args.cv_folds,
            'metric': args.metric,
            'best_hyperparameters': best_params,
            'best_optuna_value': best_value if not args.skip_optuna else None,
            'optuna_skipped': args.skip_optuna,
            'test_metrics': test_metrics,  # None if test_size=0
            'cv_metrics': cv_results if task == 'classification' else {
                'r2_mean': cv_results['r2_mean'],
                'r2_std': cv_results['r2_std'],
                'mae_mean': cv_results['mae_mean'],
                'mae_std': cv_results['mae_std'],
                'rmse_mean': cv_results['rmse_mean'],
                'rmse_std': cv_results['rmse_std'],
                'pearson_r_mean': cv_results['pearson_r_mean'],
                'pearson_r_std': cv_results['pearson_r_std'],
                'spearman_rho_mean': cv_results['spearman_rho_mean'],
                'spearman_rho_std': cv_results['spearman_rho_std']
            },
            'model_info': {
                'n_estimators': history['n_estimators'],
                'model_path': model_path
            }
        }
    
        results_path = os.path.join(args.output_dir, 'training_results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"   Results saved to {results_path}")
    
        # Final summary
        print("\n" + "=" * 80)
        print("Training Complete!")
        print("=" * 80)
        print(f"\nBest hyperparameters: {hyperparams_path}")
        print(f"Model: {model_path}")
        print(f"Results: {results_path}")
        print(f"Feature importance: {importance_path}")
        print(f"\nFinal Performance:")
        if test_metrics:
            if task == 'regression':
                print(f"  Test R²: {test_metrics['r2']:.4f}")
                print(f"  Test Pearson r: {test_metrics['pearson_r']:.4f}")
                print(f"  Test Spearman ρ: {test_metrics['spearman_rho']:.4f}")
                print(f"  Test RMSE: {test_metrics['rmse']:.4f}")
            else:  # classification
                print(f"  Test Accuracy: {test_metrics['accuracy']:.4f}")
                print(f"  Test F1: {test_metrics['f1']:.4f}")
                if 'roc_auc' in test_metrics:
                    print(f"  Test ROC-AUC: {test_metrics['roc_auc']:.4f}")
    
        if task == 'regression':
            print(f"  CV R²: {cv_results['r2_mean']:.4f} ± {cv_results['r2_std']:.4f}")
            print(f"  CV Pearson r: {cv_results['pearson_r_mean']:.4f} ± {cv_results['pearson_r_std']:.4f}")
            print(f"  CV Spearman ρ: {cv_results['spearman_rho_mean']:.4f} ± {cv_results['spearman_rho_std']:.4f}")
            print(f"  CV RMSE: {cv_results['rmse_mean']:.4f} ± {cv_results['rmse_std']:.4f}")
        else:  # classification
            print(f"  CV Accuracy: {cv_results['accuracy_mean']:.4f} ± {cv_results['accuracy_std']:.4f}")
            print(f"  CV F1: {cv_results['f1_mean']:.4f} ± {cv_results['f1_std']:.4f}")
            if 'roc_auc_mean' in cv_results:
                print(f"  CV ROC-AUC: {cv_results['roc_auc_mean']:.4f} ± {cv_results['roc_auc_std']:.4f}")
        print("=" * 80)


if __name__ == "__main__":
    main()

