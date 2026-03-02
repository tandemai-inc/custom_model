"""
Test script to verify the XGBoost training environment is set up correctly.

Tests:
1. Package imports
2. Feature calculation
3. Optuna optimization
4. XGBoost training
5. End-to-end pipeline
"""

import sys
import os

# Add parent directory to path for reduce_mordred_features
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("=" * 80)
print("XGBoost Training Environment Test")
print("=" * 80)

# Test 1: Package imports
print("\n1. Testing package imports...")
try:
    import pandas as pd
    print("   ✓ pandas")
except ImportError as e:
    print(f"   ✗ pandas: {e}")
    sys.exit(1)

try:
    import numpy as np
    print("   ✓ numpy")
except ImportError as e:
    print(f"   ✗ numpy: {e}")
    sys.exit(1)

try:
    import sklearn
    print("   ✓ scikit-learn")
except ImportError as e:
    print(f"   ✗ scikit-learn: {e}")
    sys.exit(1)

try:
    import xgboost as xgb
    print("   ✓ xgboost")
except ImportError as e:
    print(f"   ✗ xgboost: {e}")
    sys.exit(1)

try:
    import optuna
    print("   ✓ optuna")
except ImportError as e:
    print(f"   ✗ optuna: {e}")
    sys.exit(1)

try:
    from rdkit import Chem
    from rdkit.Chem import rdMolDescriptors
    from rdkit.Chem import MACCSkeys
    print("   ✓ rdkit")
except ImportError as e:
    print(f"   ✗ rdkit: {e}")
    sys.exit(1)

try:
    from mordred import Calculator, descriptors
    print("   ✓ mordred")
except ImportError as e:
    print(f"   ✗ mordred: {e}")
    sys.exit(1)

try:
    from map4 import MAP4
    print("   ✓ map4")
except ImportError as e:
    print(f"   ✗ map4: {e}")
    sys.exit(1)

# Test 2: Module imports
print("\n2. Testing local module imports...")
try:
    from featurization import (
        calculate_mordred_features,
        calculate_morgan_fingerprints,
        calculate_rdkit_fingerprints,
        calculate_maccs_keys,
        calculate_map4_fingerprints,
        calculate_all_features
    )
    print("   ✓ featurization.py")
except ImportError as e:
    print(f"   ✗ featurization.py: {e}")
    sys.exit(1)

try:
    from optuna_tuning import create_objective, optimize_hyperparameters
    print("   ✓ optuna_tuning.py")
except ImportError as e:
    print(f"   ✗ optuna_tuning.py: {e}")
    sys.exit(1)

try:
    from xgboost_trainer import (
        train_xgboost,
        evaluate_model,
        cross_validate_model,
        save_model,
        load_model,
        get_feature_importance
    )
    print("   ✓ xgboost_trainer.py")
except ImportError as e:
    print(f"   ✗ xgboost_trainer.py: {e}")
    sys.exit(1)

# Test 3: Feature calculation
print("\n3. Testing feature calculation...")
test_smiles = ["CCO", "CC(=O)O", "c1ccccc1"]  # Ethanol, Acetic acid, Benzene

try:
    print("   Testing Morgan fingerprints...")
    df_morgan = calculate_morgan_fingerprints(test_smiles)
    assert df_morgan.shape[0] == len(test_smiles), "Wrong number of rows"
    assert df_morgan.shape[1] == 2048, "Wrong number of features"
    print(f"   ✓ Morgan: {df_morgan.shape}")
except Exception as e:
    print(f"   ✗ Morgan fingerprints failed: {e}")
    sys.exit(1)

try:
    print("   Testing RDKit fingerprints...")
    df_rdkit = calculate_rdkit_fingerprints(test_smiles)
    assert df_rdkit.shape[0] == len(test_smiles), "Wrong number of rows"
    assert df_rdkit.shape[1] == 2048, "Wrong number of features"
    print(f"   ✓ RDKit: {df_rdkit.shape}")
except Exception as e:
    print(f"   ✗ RDKit fingerprints failed: {e}")
    sys.exit(1)

try:
    print("   Testing MACCS keys...")
    df_maccs = calculate_maccs_keys(test_smiles)
    assert df_maccs.shape[0] == len(test_smiles), "Wrong number of rows"
    assert df_maccs.shape[1] == 167, "Wrong number of features"
    print(f"   ✓ MACCS: {df_maccs.shape}")
except Exception as e:
    print(f"   ✗ MACCS keys failed: {e}")
    sys.exit(1)

try:
    print("   Testing MAP4 fingerprints...")
    df_map4 = calculate_map4_fingerprints(test_smiles, dimensions=1024)
    if not df_map4.empty:
        assert df_map4.shape[0] == len(test_smiles), "Wrong number of rows"
        assert df_map4.shape[1] == 1024, "Wrong number of features"
        print(f"   ✓ MAP4: {df_map4.shape}")
    else:
        print(f"   ⚠ MAP4: Empty (may not be available)")
except Exception as e:
    print(f"   ⚠ MAP4 fingerprints skipped: {e}")

# Test 4: Reduced Mordred features (if available)
print("\n4. Testing reduced Mordred features...")
try:
    # Try current directory first, then parent directory for backward compatibility
    current_path = "reduced_mordred_features.json"
    parent_path = "../reduced_mordred_features.json"
    
    reduced_features_path = None
    if os.path.exists(current_path):
        reduced_features_path = current_path
    elif os.path.exists(parent_path):
        reduced_features_path = parent_path
    
    if reduced_features_path:
        df_mordred = calculate_mordred_features(test_smiles, reduced_features_path)
        assert df_mordred.shape[0] == len(test_smiles), "Wrong number of rows"
        print(f"   ✓ Mordred: {df_mordred.shape}")
    else:
        print(f"   ⚠ Skipped (file not found: {current_path} or {parent_path})")
except Exception as e:
    print(f"   ⚠ Mordred features skipped: {e}")

# Test 5: XGBoost model creation
print("\n5. Testing XGBoost model creation...")
try:
    # Create dummy data
    X = pd.DataFrame(np.random.rand(10, 5))
    y = pd.Series(np.random.rand(10))
    
    model = xgb.XGBRegressor(n_estimators=10, random_state=42)
    model.fit(X, y)
    predictions = model.predict(X)
    
    assert len(predictions) == len(y), "Prediction length mismatch"
    print("   ✓ XGBoost model creation and prediction")
except Exception as e:
    print(f"   ✗ XGBoost failed: {e}")
    sys.exit(1)

# Test 6: Optuna study creation
print("\n6. Testing Optuna study creation...")
try:
    study = optuna.create_study(direction='maximize')
    print("   ✓ Optuna study creation")
except Exception as e:
    print(f"   ✗ Optuna failed: {e}")
    sys.exit(1)

# Test 7: End-to-end mini pipeline
print("\n7. Testing end-to-end mini pipeline...")
try:
    # Create synthetic data
    n_samples = 20
    n_features = 10
    
    X = pd.DataFrame(np.random.rand(n_samples, n_features))
    y = pd.Series(np.random.rand(n_samples) * 10 + 2)  # Range similar to real data
    
    # Split data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    # Train model
    params = {
        'n_estimators': 10,
        'max_depth': 3,
        'learning_rate': 0.1,
        'random_state': 42
    }
    
    model, _ = train_xgboost(X_train, y_train, X_test, y_test, params, verbose=False)
    
    # Evaluate
    metrics = evaluate_model(model, X_test, y_test)
    assert 'r2' in metrics, "R² metric missing"
    assert 'mae' in metrics, "MAE metric missing"
    assert 'rmse' in metrics, "RMSE metric missing"
    
    print(f"   ✓ End-to-end pipeline")
    print(f"     Test R²: {metrics['r2']:.4f}")
    print(f"     Test MAE: {metrics['mae']:.4f}")
    print(f"     Test RMSE: {metrics['rmse']:.4f}")
except Exception as e:
    print(f"   ✗ End-to-end pipeline failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 8: Model persistence
print("\n8. Testing model persistence...")
try:
    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp:
        tmp_path = tmp.name
    
    # Save model
    save_model(model, tmp_path)
    
    # Load model
    loaded_model = load_model(tmp_path)
    
    # Verify predictions match
    pred_original = model.predict(X_test)
    pred_loaded = loaded_model.predict(X_test)
    
    assert np.allclose(pred_original, pred_loaded), "Predictions don't match after load"
    print("   ✓ Model save/load")
    
    # Cleanup
    os.unlink(tmp_path)
except Exception as e:
    print(f"   ✗ Model persistence failed: {e}")
    sys.exit(1)

# Test 9: Feature importance
print("\n9. Testing feature importance extraction...")
try:
    feature_names = [f'feature_{i}' for i in range(n_features)]
    df_importance, df_importance_top = get_feature_importance(
        model, feature_names, top_n=5
    )
    
    assert len(df_importance) == n_features, "Wrong number of features in importance"
    assert len(df_importance_top) == 5, "Wrong number of top features"
    print(f"   ✓ Feature importance extraction")
except Exception as e:
    print(f"   ✗ Feature importance failed: {e}")
    sys.exit(1)

# Final summary
print("\n" + "=" * 80)
print("All tests passed! ✓")
print("=" * 80)
print("\nEnvironment is ready to use.")
print("\nTo activate the environment:")
print("  conda activate xgboost_training")
print("\nTo run training:")
print("  python train_model.py --data ../train_val_features.csv --n_trials 10")
print("=" * 80)

