import torch
import torch.nn as nn
import torch.optim as optim

from model import create_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = create_model(num_classes=3)
model = model.to(device)

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=0.0001
)

print("Training configuration created successfully!")
print("Device:", device)
print("Loss function:", criterion)
print("Optimizer:", optimizer)
print("Learning rate:", optimizer.param_groups[0]["lr"])
