from dagster import job
from .ops import pull_dvc_op, trigger_kaggle_op, collect_result_op, upload_model_to_b2_op

@job
def pipeline():
    pull_dvc_op()
    trigger_kaggle_op()
    model = collect_result_op()
    upload_model_to_b2_op(model)