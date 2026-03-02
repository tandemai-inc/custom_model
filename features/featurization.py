"""
Feature calculation module for molecular descriptors and fingerprints.

Combines:
- Reduced Mordred descriptors
- Morgan fingerprints
- RDKit fingerprints (Atom Pair)
- MACCS keys
- MAP4 fingerprints
"""

import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import MACCSkeys
import sys
import os

# Try to import reduce_mordred_features from current directory first, then parent
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

# Add current directory to path (for self-contained deployment)
sys.path.insert(0, current_dir)
# Also add parent directory for backward compatibility
sys.path.append(parent_dir)

try:
    from reduce_mordred_features import create_reduced_calculator, load_reduced_feature_list
except ImportError:
    print("Warning: Could not import reduce_mordred_features. Ensure it's in the same directory or parent directory.")
    create_reduced_calculator = None
    load_reduced_feature_list = None

try:
    from map4 import MAP4
    MAP4_AVAILABLE = True
except ImportError:
    print("Warning: MAP4 package not available. Install with: pip install map4")
    MAP4_AVAILABLE = False


def calculate_mordred_features(smiles_list, reduced_features_path=None):
    """
    Calculate reduced Mordred features.
    
    Args:
        smiles_list: List of SMILES strings
        reduced_features_path: Path to reduced_mordred_features.json
        
    Returns:
        DataFrame with Mordred features
    """
    if create_reduced_calculator is None or load_reduced_feature_list is None:
        raise ImportError("reduce_mordred_features module not available")
    
    # Load reduced feature list
    if reduced_features_path is None:
        # Try to find in current directory first, then parent directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        current_path = os.path.join(current_dir, "reduced_mordred_features.json")
        if os.path.exists(current_path):
            reduced_features_path = current_path
        else:
            parent_dir = os.path.dirname(current_dir)
            reduced_features_path = os.path.join(parent_dir, "reduced_mordred_features.json")
    
    try:
        reduced_features = load_reduced_feature_list(reduced_features_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Reduced features file not found: {reduced_features_path}")
    
    # Create reduced calculator
    calc, _ = create_reduced_calculator(reduced_features)
    
    # Calculate features
    results = []
    for smiles in smiles_list:
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                results.append({})
                continue
            
            desc_values = calc(mol)
            desc_dict = {}
            for desc, value in desc_values.items():
                if isinstance(value, Exception):
                    desc_dict[f"mordred_{desc}"] = np.nan
                else:
                    desc_dict[f"mordred_{desc}"] = value
            
            results.append(desc_dict)
        except Exception as e:
            print(f"Error processing {smiles} for Mordred: {e}")
            results.append({})
    
    return pd.DataFrame(results)


def calculate_morgan_fingerprints(smiles_list, radius=2, n_bits=2048):
    """
    Calculate Morgan (circular) fingerprints.
    
    Args:
        smiles_list: List of SMILES strings
        radius: Morgan fingerprint radius (default: 2)
        n_bits: Number of bits (default: 2048)
        
    Returns:
        DataFrame with Morgan fingerprints
    """
    results = []
    
    for smiles in smiles_list:
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                results.append({})
                continue
            
            fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits)
            
            # Convert to dictionary
            fp_dict = {}
            for i in range(n_bits):
                fp_dict[f"morgan_{i}"] = int(fp[i])
            
            results.append(fp_dict)
        except Exception as e:
            print(f"Error processing {smiles} for Morgan: {e}")
            results.append({})
    
    return pd.DataFrame(results)


def calculate_rdkit_fingerprints(smiles_list, n_bits=2048):
    """
    Calculate RDKit Atom Pair fingerprints.
    
    Args:
        smiles_list: List of SMILES strings
        n_bits: Number of bits (default: 2048)
        
    Returns:
        DataFrame with RDKit fingerprints
    """
    results = []
    
    for smiles in smiles_list:
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                results.append({})
                continue
            
            # Calculate Atom Pair fingerprint
            fp = rdMolDescriptors.GetHashedAtomPairFingerprintAsBitVect(mol, nBits=n_bits)
            
            # Convert to dictionary
            fp_dict = {}
            for i in range(n_bits):
                fp_dict[f"rdkit_{i}"] = int(fp[i])
            
            results.append(fp_dict)
        except Exception as e:
            print(f"Error processing {smiles} for RDKit: {e}")
            results.append({})
    
    return pd.DataFrame(results)


def calculate_maccs_keys(smiles_list):
    """
    Calculate MACCS keys.
    
    Args:
        smiles_list: List of SMILES strings
        
    Returns:
        DataFrame with MACCS keys
    """
    results = []
    n_bits = 167  # MACCS has 167 bits
    
    for smiles in smiles_list:
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                results.append({})
                continue
            
            fp = MACCSkeys.GenMACCSKeys(mol)
            
            # Convert to dictionary
            fp_dict = {}
            for i in range(n_bits):
                fp_dict[f"maccs_{i}"] = int(fp[i])
            
            results.append(fp_dict)
        except Exception as e:
            print(f"Error processing {smiles} for MACCS: {e}")
            results.append({})
    
    return pd.DataFrame(results)


def calculate_map4_fingerprints(smiles_list, dimensions=1024, radius=1):
    """
    Calculate MAP4 (Molecular Accessible Pathways 4) fingerprints.
    
    Args:
        smiles_list: List of SMILES strings
        dimensions: Number of dimensions (default: 1024)
        radius: MAP4 radius (default: 1)
        
    Returns:
        DataFrame with MAP4 fingerprints
    """
    if not MAP4_AVAILABLE:
        print("Warning: MAP4 not available. Returning empty DataFrame.")
        return pd.DataFrame()
    
    results = []
    map4_calc = MAP4(dimensions=dimensions, radius=radius)
    
    for smiles in smiles_list:
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                results.append({})
                continue
            
            # Calculate MAP4 fingerprint
            fp = map4_calc.calculate(mol)
            
            # Convert to dictionary
            fp_dict = {}
            for i, val in enumerate(fp):
                fp_dict[f"map4_{i}"] = float(val)
            
            results.append(fp_dict)
        except Exception as e:
            print(f"Error processing {smiles} for MAP4: {e}")
            results.append({})
    
    return pd.DataFrame(results)


def calculate_all_features(smiles_list, reduced_features_path=None, 
                           include_map4=True, map4_dimensions=1024):
    """
    Calculate and combine all molecular features.
    
    Args:
        smiles_list: List of SMILES strings
        reduced_features_path: Path to reduced_mordred_features.json
        include_map4: Whether to include MAP4 fingerprints (default: True)
        map4_dimensions: MAP4 fingerprint dimensions (default: 1024)
        
    Returns:
        DataFrame with all combined features
    """
    print("Calculating features...")
    print(f"  Processing {len(smiles_list)} molecules")
    
    # Calculate each feature type
    print("  Calculating Mordred features...")
    df_mordred = calculate_mordred_features(smiles_list, reduced_features_path)
    
    print("  Calculating Morgan fingerprints...")
    df_morgan = calculate_morgan_fingerprints(smiles_list)
    
    print("  Calculating RDKit fingerprints...")
    df_rdkit = calculate_rdkit_fingerprints(smiles_list)
    
    print("  Calculating MACCS keys...")
    df_maccs = calculate_maccs_keys(smiles_list)
    
    df_map4 = pd.DataFrame()
    if include_map4:
        print("  Calculating MAP4 fingerprints...")
        df_map4 = calculate_map4_fingerprints(smiles_list, dimensions=map4_dimensions)
    
    # Combine all features
    print("  Combining features...")
    dfs = [df_mordred, df_morgan, df_rdkit, df_maccs]
    if not df_map4.empty:
        dfs.append(df_map4)
    
    # Concatenate horizontally
    df_combined = pd.concat(dfs, axis=1)
    
    # Convert all columns to numeric (handles string/object types)
    print("  Converting features to numeric types...")
    for col in df_combined.columns:
        df_combined[col] = pd.to_numeric(df_combined[col], errors='coerce')
    
    # Fill NaN values with 0
    df_combined = df_combined.fillna(0)
    
    # Ensure all columns are numeric (convert any remaining object types)
    df_combined = df_combined.astype(float)
    
    print(f"  Total features: {df_combined.shape[1]}")
    print(f"  Feature breakdown:")
    print(f"    Mordred: {df_mordred.shape[1]}")
    print(f"    Morgan: {df_morgan.shape[1]}")
    print(f"    RDKit: {df_rdkit.shape[1]}")
    print(f"    MACCS: {df_maccs.shape[1]}")
    if not df_map4.empty:
        print(f"    MAP4: {df_map4.shape[1]}")
    
    return df_combined

