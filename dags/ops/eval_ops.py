import mlflow
import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from dagster import op

@op
def evaluate_model(model_path: str, val_dir: str):
    # Dataset dummy de validación
    X_val = np.random.rand(50, 10)
    y_val = np.random.randint(0, 2, size=(50,))
    dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32),
                            torch.tensor(y_val, dtype=torch.long))
    loader = DataLoader(dataset, batch_size=16)

    model = torch.nn.Sequential(
        torch.nn.Linear(10, 32),
        torch.nn.ReLU(),
        torch.nn.Linear(32, 2)
    )
    model.load_state_dict(torch.load(model_path))
    model.eval()

    correct, total = 0, 0
    with torch.no_grad():
        for X_batch, y_batch in loader:
            outputs = model(X_batch)
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == y_batch).sum().item()
            total += y_batch.size(0)

    accuracy = correct / total

    with mlflow.start_run(nested=True):  # nested run bajo el entrenamiento
        mlflow.log_metric("accuracy", accuracy)

    print(f"Evaluación final - Accuracy: {accuracy}")
    return {"accuracy": accuracy}
