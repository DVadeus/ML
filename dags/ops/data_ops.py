from pathlib import Path
import zipfile
import subprocess
from dagster import op

DATA_DIR = Path("data")

@op
def pull_dvc_data():
    subprocess.run(["dvc", "pull"], check = True)
    return str(DATA_DIR)

@op
def extract_train_data(base_dir:str):
    zip_path = Path(base_dir) / "train.zip"
    extract_path = Path(base_dir) / "train"
    if not extract_path.exists():
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_path)
    return str(extract_path)

@op
def extract_val_data(base_dir: str):
    zip_path = Path(base_dir) / "val.zip"
    extract_path = Path(base_dir) / "val"
    if not extract_path.exists():
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_path)
    return str(extract_path)