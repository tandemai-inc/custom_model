"""Error codes for XGBoost workflow."""

from ta_workflow.utils.ta_workflow_exception import TAWorkflowException


class XGBoostWorkFlowErrors:
    """Error codes for XGBoost workflow."""

    XGBOOST_CONFIG_ERROR = 1001
    XGBOOST_INPUT_FILE_NOT_EXIST = 1002
    XGBOOST_MODEL_DIR_NOT_EXIST = 1003
    XGBOOST_OUTPUT_DIR_ERROR = 1004


def raise_config_error(message: str):
    """Raise config error."""
    raise TAWorkflowException(
        XGBoostWorkFlowErrors.XGBOOST_CONFIG_ERROR,
        message,
    )


def raise_input_not_found(filepath: str):
    """Raise input file not found."""
    raise TAWorkflowException(
        XGBoostWorkFlowErrors.XGBOOST_INPUT_FILE_NOT_EXIST,
        f"Input file not found: {filepath}",
    )


def raise_model_dir_not_found(model_dir: str):
    """Raise model dir not found."""
    raise TAWorkflowException(
        XGBoostWorkFlowErrors.XGBOOST_MODEL_DIR_NOT_EXIST,
        f"Model directory not found: {model_dir}",
    )
