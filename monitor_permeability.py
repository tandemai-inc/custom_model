#!/usr/bin/env python3
"""
Monitor permeability training progress.
Can be run periodically via cron or manually.
"""

import os
import json
import subprocess
from datetime import datetime
from pathlib import Path

def check_process_running():
    """Check if training process is running."""
    try:
        result = subprocess.run(
            ['ps', 'aux'], 
            capture_output=True, 
            text=True
        )
        # Check for permeability training
        return 'train_model.py' in result.stdout and 'permeability' in result.stdout
    except:
        return False

def check_results_directory(output_dir='permeability_with_embeddings'):
    """Check what files exist in results directory."""
    if not os.path.exists(output_dir):
        return None, []
    
    files = []
    for f in os.listdir(output_dir):
        filepath = os.path.join(output_dir, f)
        if os.path.isfile(filepath):
            size = os.path.getsize(filepath)
            mtime = os.path.getmtime(filepath)
            files.append({
                'name': f,
                'size': size,
                'modified': datetime.fromtimestamp(mtime)
            })
    
    return output_dir, sorted(files, key=lambda x: x['modified'], reverse=True)

def format_size(size_bytes):
    """Format file size."""
    if size_bytes >= 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.2f} MB"
    elif size_bytes >= 1024:
        return f"{size_bytes / 1024:.2f} KB"
    else:
        return f"{size_bytes} B"

def main():
    output_dir = os.environ.get('TRAINING_OUTPUT_DIR', 'permeability_with_embeddings')
    
    print("=" * 80)
    print(f"PERMEABILITY TRAINING PROGRESS MONITOR")
    print("=" * 80)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Output directory: {output_dir}")
    print()
    
    # Check if process is running
    is_running = check_process_running()
    print(f"Training process running: {'✓ Yes' if is_running else '✗ No'}")
    
    if is_running:
        try:
            result = subprocess.run(
                ['ps', 'aux'], 
                capture_output=True, 
                text=True
            )
            lines = [l for l in result.stdout.split('\n') if 'train_model.py' in l and 'permeability' in l and 'grep' not in l]
            if lines:
                parts = lines[0].split()
                if len(parts) >= 3:
                    cpu = parts[2]
                    mem = float(parts[5]) / 1024
                    print(f"  CPU usage: {cpu}%")
                    print(f"  Memory: {mem:.1f} MB")
        except:
            pass
    
    print()
    
    # Check results directory
    results_dir, files = check_results_directory(output_dir)
    
    if results_dir:
        print(f"Results directory: {results_dir}")
        print(f"Files found: {len(files)}")
        
        if files:
            print("\nRecent files (last 5):")
            for f in files[:5]:
                age = (datetime.now() - f['modified']).total_seconds() / 60
                print(f"  {f['name']:50s} {format_size(f['size']):>10s}  {age:>6.1f} min ago")
        
        # Check completion status
        file_names = [f['name'] for f in files]
        
        if 'training_results.json' in file_names:
            print("\n" + "=" * 80)
            print("🎉 TRAINING COMPLETE! 🎉")
            print("=" * 80)
            try:
                with open(os.path.join(results_dir, 'training_results.json'), 'r') as f:
                    results = json.load(f)
                
                print(f"\nDataset: {results.get('data_file', 'N/A')}")
                print(f"Task: {results.get('task', 'N/A')}")
                print(f"Samples: {results.get('n_samples', 'N/A'):,}")
                print(f"Features: {results.get('n_features', 'N/A'):,}")
                
                if 'cv_metrics' in results:
                    cv = results['cv_metrics']
                    if 'r2_mean' in cv:
                        print(f"\nCross-Validation Results:")
                        print(f"  R²: {cv['r2_mean']:.4f} ± {cv['r2_std']:.4f}")
                        print(f"  MAE: {cv['mae_mean']:.4f} ± {cv['mae_std']:.4f}")
                        print(f"  RMSE: {cv['rmse_mean']:.4f} ± {cv['rmse_std']:.4f}")
                        print(f"  Pearson r: {cv['pearson_r_mean']:.4f} ± {cv['pearson_r_std']:.4f}")
                        print(f"  Spearman ρ: {cv['spearman_rho_mean']:.4f} ± {cv['spearman_rho_std']:.4f}")
            except Exception as e:
                print(f"Error reading results: {e}")
                
        elif 'best_model.pkl' in file_names:
            print("\n⏱️  Status: Finalizing results...")
            latest = max(f['modified'] for f in files if f['name'] == 'best_model.pkl')
            age = (datetime.now() - latest).total_seconds() / 60
            print(f"   Model saved {age:.1f} minutes ago")
            
        elif 'best_hyperparameters.json' in file_names:
            print("\n⏱️  Status: Training final model...")
            latest = max(f['modified'] for f in files if f['name'] == 'best_hyperparameters.json')
            age = (datetime.now() - latest).total_seconds() / 60
            print(f"   Hyperparameters saved {age:.1f} minutes ago")
            print(f"   Estimated time remaining: 10-20 minutes")
            
        elif 'selected_features.json' in file_names:
            print("\n⏱️  Status: Optuna optimization in progress...")
            latest = max(f['modified'] for f in files if f['name'] == 'selected_features.json')
            age = (datetime.now() - latest).total_seconds() / 60
            print(f"   Feature selection completed {age:.1f} minutes ago")
            print(f"   Estimated time remaining: 1-3 hours (50 trials × 5-fold CV)")
            
        else:
            print("\n⏱️  Status: Feature selection in progress...")
            if files:
                latest = max(f['modified'] for f in files)
                age = (datetime.now() - latest).total_seconds() / 60
                print(f"   Last file updated {age:.1f} minutes ago")
            print(f"   Estimated time remaining: 10-30 minutes for feature selection")
    else:
        print("\n⏳ Results directory not created yet - training initializing...")
        print("   This stage typically takes 5-15 minutes")
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()

