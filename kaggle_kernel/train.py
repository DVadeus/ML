import os
import mlflow
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

# --- Variables de entorno ---#
# MLFlow
os.environ.setdefault("MLFLOW_TRACKING_URI", os.getenv("MLFLOW_TRACKING_URI",""))
os.environ.setdefault("MLFLOW_TRACKING_USERNAME", os.getenv("MLFLOW_TRACKING_USERNAME",""))
os.environ.setdefault("MLFLOW_TRACKING_PASSWORD", os.getenv("MLFLOW_TRACKING_PASSWORD",""))
# Backblaze S3
os.environ["AWS_ENDPOINT_URL"] = os.getenv("B2_ENDPOINT","")
os.environ["AWS_ACCESS_KEY_ID"] = os.getenv("B2_KEY_ID","")
os.environ["AWS_SECRET_ACCESS_KEY"] = os.getenv("B2_API_KEY","")

mlflow.set_tracking_uri("https://dagshub.com/DVadeus/ML.mlflow")

def get_data():
    #TODO
    #Coco detection o loader
    #En producción reemplaza get_dummy_data() por el CocoDetection loader apuntando a los archivos descomprimidos.
    return None


def train_and_log():

    mlflow.set_experiment("coco")
    lr = 0.01
    batch_size = 16
    epochs = 5

    X = np.random.rand(100, 10)
    y = np.random.randint(0, 2, size = (100,))
    dataset = TensorDataset(torch.tensor(X, dtype=torch.float32),
                            torch.tensor(y, dtype=torch.long))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = nn.Sequential(
        nn.Linear(10, 32),
        nn.ReLU(),
        nn.Linear(32,2)
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = lr)

    mlflow.set_experiment("coco-training")

    with mlflow.start_run():
        mlflow.log_param("lr", lr)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("epochs", epochs)

        for epoch in range(epochs):
            total_loss = 0
            for X_batch, y_batch in loader:
                optimizer.zero_grad()
                output = model(X_batch)
                loss = criterion(output, y_batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            avg_loss = total_loss / len(loader)
            mlflow.log_metric("loss", avg_loss, step=epoch)
            print(f"Epoch {epoch+1}, Loss: {avg_loss}")

        os.makedirs("output", exist_ok=True)
        model_path = "output/model.pth"
        torch.save(model.state_dict(), model_path)

        #Guardando el modelo en MLFlow
        mlflow.log_artifact(model_path)

        #Guardando modelo en DVC
        #TODO
        # Qué sucede si creo varios modelos con el mismo nombre en DVC
        #os.system("dvc add output/model.pth")
        #os.system("git add output/model.pth.dvc && git commit -m 'model from kaggle' && git push")
        #os.system("dvc push")
        # Es mejor guardar en DVC o en Dagshub para luego desplegar?
    
if __name__ == "__main__":
    train_and_log()





