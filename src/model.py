import torch
import torch.nn as nn
from torchvision import models

def create_model(num_classes=3):
    weights = models.ResNet18_Weights.DEFAULT

    model = models.resnet18(weights=weights)

    input_features = model.fc.in_features

    model.fc = nn.Linear(input_features, num_classes)

    return model


if __name__ == "__main__":
    model = create_model(num_classes=3)

    print("ResNet-18 model created successfully!")
    print("Number of output classes:", model.fc.out_features)

    sample_input = torch.randn(1, 3, 224, 224)

    output = model(sample_input)

    print("Output shape:", output.shape)
