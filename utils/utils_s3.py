import boto3
import os

def get_s3_client():
    return boto3.client(
        "s3",
        endpoint_url = os.getenv("B2_ENDPOINT"),
        aws_access_key_id = os.getenv("B2_KEY_ID"),
        aws_secret_access_key = os.getenv("B2_API_KEY")
    )

def download_from_s3(bucket, key, local_path):
    s3 = get_s3_client()
    s3.download_file(bucket, key, local_path)
    return local_path

def upload_to_s3(bucket, key, local_path):
    s3 = get_s3_client()
    s3.upload_file(local_path, bucket, key)
    return f"s3://{bucket}/{key}"

