import subprocess, os
from dagster import op
from pathlib import Path
import boto3

@op
def pull_dvc_op(context):
    context.log.info("Running dvc pull")
    subprocess.run(["dvc", "pull"], check=True)
    return "data pulled" 

@op
def trigger_kaggle_op(context):
    context.log.info("Triggering Kaggle Kernel")
    subprocess.run(["python", "scripts/trigger_kaggle.py"], check=True)
    return "kaggle triggered"

@op
def collect_result_op(context):
    out_dir = Path("artifacts/kaggle_output")
    model_files = list(out_dir.rglob("model.pth"))
    if not model_files:
        raise Exception("No model.pth found in Kaggle output")
    model_file = model_files[0]
    dest = Path("artifacts/model_from_kaggle.pth")
    dest.parent.mkdir(parents=True, exist_ok=True)
    model_file.replace(dest)
    context.log.info(f"Model stored at {dest}")
    return str(dest)

@op
def upload_model_to_b2_op(context, model_path: str):
    s3 = boto3.client(
        "s3",
        endpoint_url = os.getenv("B2_ENDPOINT"),
        aws_access_key_id = os.getenv("B2_API_KEY"),
        aws_secret_access_key = os.getenv("B2_API_KEY")
    )
    bucket = "bucketB2DV"
    key = f"models/{Path(model_path).name}"
    context.log.info(f"Uploading {model_path} to {bucket}/{key}")
    s3.upload_file(model_path, bucket, key)
    return f"s3://{bucket}/{key}"