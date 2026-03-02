#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
XGBoost training/prediction workflow master script.
Uses ta_workflow for job scheduling and monitoring, following the ADMET pattern.
"""

import argparse
import json
import os
import pathlib

import yaml
from ta_base.web_api.service import WebService
from ta_workflow.utils.ta_workflow_exception import TAWorkflowException
from ta_workflow.workflow.workflow_manager import WorkflowManager

from workflows.utils.errors import (
    XGBoostWorkFlowErrors,
    raise_config_error,
    raise_input_not_found,
    raise_model_dir_not_found,
)

# Repo root (xgboost_training directory)
PKG_DIR = pathlib.Path(__file__).resolve().parent.parent

# Version for workflow (use placeholder if no package version)
try:
    import xgboost_training
    XGBOOST_VERSION = getattr(xgboost_training, "__version__", "0.1.0")
except ImportError:
    XGBOOST_VERSION = "0.1.0"


class TaXGBoostWorkflow(WorkflowManager):
    """XGBoost training/prediction workflow using ta_workflow."""

    def __init__(self, config, working_dir):
        super(TaXGBoostWorkflow, self).__init__(config, working_dir)
        self.config = config
        self.store.debug = self.args.common.get("debug", False)
        self.store.working_dir = working_dir
        self.store.root_dir = os.path.join(working_dir, "scr")
        self.store.job_type = self.args.common.get("job_type", "slurm")
        self.current_working_dir = self.store.root_dir
        self.web_service = WebService()
        self.store.log = os.path.join(self.store.root_dir, "logs", "xgboost_workflow.log")
        self.task_id = self.workflow_id
        self.mode = None  # "training" or "prediction"
        self.output_file = None

    def __validate_config(self):
        """Validate workflow config."""
        # Support both xgboost_training and xgboost_prediction parameter keys
        train_params = self.args.parameter.get("xgboost_training", {})
        pred_params = self.args.parameter.get("xgboost_prediction", {})

        if train_params:
            self.mode = "training"
            para = train_params
            if not para.get("input_data"):
                raise_config_error("xgboost_training requires 'input_data'")
            if not os.path.exists(para["input_data"]):
                raise_input_not_found(para["input_data"])
            output_dir = para.get("output_dir", os.path.join(self.store.working_dir, "results"))
            para["output_dir"] = output_dir
            self.output_file = os.path.join(output_dir, "training_results.json")
        elif pred_params:
            self.mode = "prediction"
            para = pred_params
            if not para.get("input_data"):
                raise_config_error("xgboost_prediction requires 'input_data'")
            if not para.get("model_dir"):
                raise_config_error("xgboost_prediction requires 'model_dir'")
            if not os.path.exists(para["input_data"]):
                raise_input_not_found(para["input_data"])
            if not os.path.exists(para["model_dir"]):
                raise_model_dir_not_found(para["model_dir"])
            output_file = para.get("output_file")
            if not output_file:
                output_file = os.path.join(self.store.working_dir, "predictions.csv")
            para["output_file"] = output_file
            self.output_file = output_file
        else:
            raise_config_error(
                "Config must have 'xgboost_training' or 'xgboost_prediction' under parameter"
            )

    def create_tasks(self):
        """Create workflow tasks."""
        if self.mode == "training":
            para = self.args.parameter.get("xgboost_training")
        else:
            para = self.args.parameter.get("xgboost_prediction")

        # Build params for execute script
        params = {
            "mode": self.mode,
            "input_data": para["input_data"],
            "output_dir": para.get("output_dir", self.store.root_dir),
        }
        if self.mode == "training":
            params.update({
                "task": para.get("task", "regression"),
                "n_trials": para.get("n_trials", 100),
                "cv_folds": para.get("cv_folds", 5),
                "test_path": para.get("test_path"),
                "skip_optuna": para.get("skip_optuna", False),
                "use_feature_selection": para.get("use_feature_selection", False),
                "feature_selection_ratio": para.get("feature_selection_ratio", 1.0),
                "calculate_confidence": para.get("calculate_confidence", False),
                "confidence_k_neighbors": para.get("confidence_k_neighbors", 5),
                "cache_features": para.get("cache_features", True),
            })
        else:
            params.update({
                "model_dir": para["model_dir"],
                "output_file": para["output_file"],
                "calculate_confidence": para.get("calculate_confidence", False),
                "smiles_col": para.get("smiles_col", "smiles"),
            })

        # Write params to file for execute script
        params_file = os.path.join(self.current_working_dir, "workflow_params.json")
        os.makedirs(self.current_working_dir, exist_ok=True)
        with open(params_file, "w") as f:
            json.dump(params, f, indent=2)

        # Add params_file to params for execute (if framework passes it)
        params["__params_file__"] = params_file

        execute_file = os.path.join(PKG_DIR, "workflows", "execute", "xgboost_execute.py")
        if not os.path.exists(execute_file):
            raise TAWorkflowException(
                XGBoostWorkFlowErrors.XGBOOST_CONFIG_ERROR,
                f"Execute file not found: {execute_file}",
            )

        slurm_para = self.get_extend_parameter("default", self.args)
        slurm_para.setdefault("cpus-per-task", 8)
        slurm_para.setdefault("mem", "16G")
        slurm_para.setdefault("time", "24:00:00")

        slurm_output = os.path.join(self.current_working_dir, "slurm-%A.out")

        job = self.create_job(
            task_id="xgboost-task",
            log_path=slurm_output,
            extend_parameter=slurm_para,
            execute_file=execute_file,
            job_name=self.workflow_id,
            parameter=params,
        )
        self.add_task(job)
        return True

    def __update_state(self, state: int, message: str = ""):
        """Update workflow state. State: 1=queue, 2=starting, 3=failed, 4=completed, 9=cancelled."""
        self.logger.info("Updating state")
        body = {
            "workFolder": self.store.working_dir,
            "moduleType": "xgboost",
            "status": state,
            "message": message,
        }
        self.logger.info(f"Update state - request body: {body}")
        response = self.web_service.update_state(body)
        self.logger.info(f"Update state - response: {response}")

    def __upload_errors(self):
        """Upload errors to viz."""
        self.logger.info("Uploading errors")
        upload_errors = self.get_output_errors()
        if not upload_errors:
            self.logger.info("No error to upload!")
            return
        body = {
            "workFolder": self.store.working_dir,
            "moduleType": "xgboost",
            "errors": json.dumps(upload_errors),
        }
        self.logger.info(f"Upload errors - request body: {body}")
        response = self.web_service.upload_errors(body)
        self.logger.info(f"Upload errors - response: {response}")

    def __upload_result(self):
        """Upload result to viz."""
        self.logger.info("Uploading result")
        result = {
            "workFolder": self.store.working_dir,
            "moduleType": "xgboost",
            "moduleResult": {
                "output": self.output_file,
                "gpu_hours": 0,
                "version": XGBOOST_VERSION,
            },
        }
        self.logger.info(f"Upload results - request body: {result}")
        response = self.web_service.upload_result(result)
        self.logger.info(f"Upload results - response: {response}")

    def init(self):
        """Create folders and validate config before workflow runs."""
        self.__update_state(2)  # starting
        if not os.path.exists(self.store.root_dir):
            os.makedirs(self.store.root_dir)
        for fname in ["job.csv", "job.json"]:
            job_summary = os.path.join(self.store.root_dir, fname)
            if os.path.exists(job_summary):
                os.remove(job_summary)
        self.__validate_config()

    def summary_tasks(self):
        """Print task summary."""
        print("Summary task @ xgboost workflow")
        summary_info = super(TaXGBoostWorkflow, self).summary_tasks()
        total_cpu_time = summary_info.get("summary", {}).get("total_cpu_time", 0)
        total_cpu_format = "%02dh:%02dm:%02ds" % (
            total_cpu_time // 3600,
            total_cpu_time % 3600 // 60,
            total_cpu_time % 60,
        )
        print(f"Total CPU task time: {total_cpu_format}")

    def post(self):
        """Post-workflow: update state, upload result and errors."""
        print("XGBoost workflow version: %s" % XGBOOST_VERSION)
        try:
            self.summary_tasks()
        finally:
            if self.failed:
                self.__update_state(3)
            else:
                self.__update_state(4)
            self.__upload_result()
            self.__upload_errors()

    def cancel_callback(self):
        """Called when workflow is cancelled."""
        try:
            self.summary_tasks()
        finally:
            self.__update_state(9)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-i", "--in_config", required=True, help="Config YAML file path")
    args = parser.parse_args()
    _config = args.in_config

    working_dir = os.path.dirname(os.path.abspath(_config))
    if not working_dir:
        working_dir = os.getcwd()

    workflow = TaXGBoostWorkflow(_config, working_dir)
    workflow.set_terminate_callback()
    workflow.main()
