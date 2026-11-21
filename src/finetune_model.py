
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path

from model import create_model
from data_loader import train_loader, val_loader

# Select device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Load the existing baseline model
model = create_model(num_classes=3)

baseline_path = Path("models/baseline_resnet18.pth")
model.load_state_dict(torch.load(baseline_path, map_location=device))

print("Baseline model loaded successfully!")

# Freeze all layers
for param in model.parameters():
    param.requires_grad = False

# Unfreeze layer4 and final classification layer
for param in model.layer4.parameters():
    param.requires_grad = True

for param in model.fc.parameters():
    param.requires_grad = True

model = model.to(device)

# Loss function
criterion = nn.CrossEntropyLoss()

# Different learning rates for different layers
optimizer = optim.Adam([
    {"params": model.layer4.parameters(), "lr": 0.00001},
    {"params": model.fc.parameters(), "lr": 0.0001}
])

EPOCHS = 3
best_accuracy = 0.0

model_dir = Path("models")
model_dir.mkdir(exist_ok=True)

# Training loop
for epoch in range(EPOCHS):

    model.eval()
    model.layer4.train()
    model.fc.train()

    train_loss = 0.0
    train_correct = 0
    train_total = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        train_loss += loss.item() * images.size(0)

        predictions = outputs.argmax(dim=1)
        train_correct += (predictions == labels).sum().item()
        train_total += labels.size(0)

    train_loss /= train_total
    train_accuracy = train_correct / train_total

    # Validation
    model.eval()

    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item() * images.size(0)

            predictions = outputs.argmax(dim=1)
            val_correct += (predictions == labels).sum().item()
            val_total += labels.size(0)

    val_loss /= val_total
    val_accuracy = val_correct / val_total

    print(f"\nEpoch {epoch + 1}/{EPOCHS}")
    print(f"Train Loss: {train_loss:.4f}")
    print(f"Train Accuracy: {train_accuracy:.4f}")
    print(f"Validation Loss: {val_loss:.4f}")
    print(f"Validation Accuracy: {val_accuracy:.4f}")

    if val_accuracy > best_accuracy:

        best_accuracy = val_accuracy

        torch.save(
            model.state_dict(),
            model_dir / "finetuned_resnet18.pth"
        )

        print("Best fine-tuned model saved!")

print("\nFine-tuning completed!")
print(f"Best validation accuracy: {best_accuracy:.4f}")
