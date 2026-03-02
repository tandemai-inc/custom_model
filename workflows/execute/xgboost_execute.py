#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Execute script for XGBoost workflow jobs.
Bridges workflow parameters to model/train_model.py or model/predict.py.

Expects params via:
  - --params_file /path/to/params.json (written by workflow)
  - Or env XGBOOST_PARAMS_FILE
  - Or workflow_params.json in current directory
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

# Repo root (parent of workflows/)
REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def load_params(params_file=None):
    """Load params from file."""
    if params_file and os.path.exists(params_file):
        with open(params_file, "r") as f:
            return json.load(f)
    # Check argv for params file path (workflow may pass as first arg)
    if len(sys.argv) > 1 and os.path.exists(sys.argv[1]):
        with open(sys.argv[1], "r") as f:
            return json.load(f)
    env_file = os.environ.get("XGBOOST_PARAMS_FILE")
    if env_file and os.path.exists(env_file):
        with open(env_file, "r") as f:
            return json.load(f)
    cwd_file = os.path.join(os.getcwd(), "workflow_params.json")
    if os.path.exists(cwd_file):
        with open(cwd_file, "r") as f:
            return json.load(f)
    raise FileNotFoundError("No params file found. Use --params_file or XGBOOST_PARAMS_FILE")


def run_training(params):
    """Run train_model.py with params."""
    cmd = [
        sys.executable,
        str(REPO_ROOT / "model" / "train_model.py"),
        "--data", params["input_data"],
        "--output_dir", params["output_dir"],
        "--task", params.get("task", "regression"),
        "--n_trials", str(params.get("n_trials", 100)),
        "--cv_folds", str(params.get("cv_folds", 5)),
    ]
    if params.get("test_path"):
        cmd.extend(["--test_path", params["test_path"]])
    if params.get("skip_optuna"):
        cmd.append("--skip_optuna")
    if params.get("use_feature_selection"):
        cmd.append("--use_feature_selection")
        if params.get("feature_selection_ratio"):
            cmd.extend(["--feature_selection_ratio", str(params["feature_selection_ratio"])])
    if params.get("calculate_confidence"):
        cmd.append("--calculate_confidence")
        if params.get("confidence_k_neighbors"):
            cmd.extend(["--confidence_k_neighbors", str(params["confidence_k_neighbors"])])
    if params.get("cache_features", True):
        cmd.append("--cache_features")
    return subprocess.run(cmd, cwd=str(REPO_ROOT))


def run_prediction(params):
    """Run predict.py with params."""
    cmd = [
        sys.executable,
        str(REPO_ROOT / "model" / "predict.py"),
        "--data", params["input_data"],
        "--model_dir", params["model_dir"],
        "--output", params["output_file"],
    ]
    if params.get("calculate_confidence"):
        cmd.append("--calculate_confidence")
    if params.get("smiles_col"):
        cmd.extend(["--smiles_col", params["smiles_col"]])
    return subprocess.run(cmd, cwd=str(REPO_ROOT))


def main():
    parser = argparse.ArgumentParser(description="XGBoost workflow execute script")
    parser.add_argument("--params_file", type=str, help="Path to params JSON file")
    args = parser.parse_args()

    params = load_params(args.params_file)
    mode = params.get("mode", "training")

    if mode == "prediction":
        return run_prediction(params).returncode
    return run_training(params).returncode


if __name__ == "__main__":
    sys.exit(main())
