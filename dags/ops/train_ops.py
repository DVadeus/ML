from dagster import op
import mlflow
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

import mlflow

mlflow.set_tracking_uri("https://dagshub.com/DVadeus/ML.mlflow")

@op
def train_model(train_dir:str):
    print(f"Entrenando modelo con datos en {train_dir}")

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

        model_path = "artifacts/model.pth"
        torch.save(model.state_dict(), model_path)
        mlflow.log_artifact(model_path)
    
    return model_path





