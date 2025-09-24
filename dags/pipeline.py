from dagster import job
from .ops.data_ops import pull_dvc_data, extract_train_data, extract_val_data
from .ops.train_ops import train_model
from .ops.eval_ops import evaluate_model

@job
def ml_pipeline():
    base_dir = pull_dvc_data()
    train_dir = extract_train_data(base_dir)
    val_dir = extract_val_data(base_dir)
    model_path = train_model(train_dir)
    evaluate_model(model_path, val_dir)