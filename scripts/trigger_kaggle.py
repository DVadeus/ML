import time, os, json
from kaggle import KaggleApi

api = KaggleApi()
api.authenticate()

KERNEL_DIR = "kaggle_kernel"
print("Pushing kernel to Kaggle")
api.kernels_push(KERNEL_DIR)

#Obtener kernel slug a partir del metadata id
meta = json.load(open(os.path.join(KERNEL_DIR, "kernel-metadata.json")))
kernel_id = meta.get("id")

def wait_for_completition(kernel_id, poll_interval=10):
    print(f"Waiting kernel {kernel_id} to finish..." )
    while True:
        status = api.kernels_status(kernel_id)
        print("Status: ",status)
        if "complete" or "finished" in str(status).lower():
            print("Kernel finished")
            break
        if "error" in str(status).lower():
            raise RuntimeError(f"Kaggle kernel failed: {status}")
        time.sleep(poll_interval)

wait_for_completition(kernel_id)

#Descargar output del kernel 
out_dir = "artifacts/kaggle_output"
os.makedirs(out_dir, exist_ok=True)
print("Downloading kernel output...")
api.kernels_output(kernel_id, path=out_dir)
print("Output downloaded to", out_dir)