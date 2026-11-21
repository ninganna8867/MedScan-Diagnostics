
import mlflow
import torch
import torch.nn as nn
import torch.optim as optim

from model import create_model
from data_loader import train_loader, val_loader

mlflow.set_experiment("MedScan_Hyperparameter_Experiments")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
criterion = nn.CrossEntropyLoss()

# Configuration 1: Existing baseline results
with mlflow.start_run(run_name="Config_1_Baseline"):
    mlflow.log_param("model", "ResNet18")
    mlflow.log_param("learning_rate", 0.0001)
    mlflow.log_param("batch_size", 32)
    mlflow.log_param("epochs", 5)
    mlflow.log_metric("validation_accuracy", 0.7533)
    mlflow.log_metric("validation_loss", 0.6606)

print("Configuration 1 baseline logged.")

# Configuration 2: Train classifier layer with a different learning rate
model = create_model(num_classes=3)
model.load_state_dict(
    torch.load("models/baseline_resnet18.pth", map_location=device)
)

for param in model.parameters():
    param.requires_grad = False

for param in model.fc.parameters():
    param.requires_grad = True

model.to(device)
optimizer = optim.Adam(model.fc.parameters(), lr=0.00001)

model.eval()
model.fc.train()

train_total = 0

for images, labels in train_loader:
    images = images.to(device)
    labels = labels.to(device)

    optimizer.zero_grad()
    outputs = model(images)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

    train_total += labels.size(0)

print("Configuration 2 training completed.")

# Validation
model.eval()
correct = 0
total = 0
total_loss = 0.0

with torch.no_grad():
    for images, labels in val_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        total_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)

accuracy = correct / total
average_loss = total_loss / total

with mlflow.start_run(run_name="Config_2_Learning_Rate_0.00001"):
    mlflow.log_param("model", "ResNet18")
    mlflow.log_param("learning_rate", 0.00001)
    mlflow.log_param("batch_size", 32)
    mlflow.log_param("epochs", 1)
    mlflow.log_metric("validation_accuracy", accuracy)
    mlflow.log_metric("validation_loss", average_loss)

torch.save(model.state_dict(), "models/experiment_config2.pth")

print("Configuration 2 validation accuracy:", round(accuracy, 4))
print("Configuration 2 validation loss:", round(average_loss, 4))
print("Both configurations logged in MLflow.")
