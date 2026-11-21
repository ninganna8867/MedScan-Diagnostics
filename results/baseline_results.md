# MedScan - Baseline Training Results

## Model
- Architecture: ResNet-18
- Approach: Transfer Learning
- Pre-trained weights: ImageNet
- Trainable layers: Final fully connected layer
- Optimizer: Adam
- Learning rate: 0.0001
- Loss function: CrossEntropyLoss
- Epochs: 5
- Batch size: 32

## Dataset
- COVID: 1000 images
- Normal: 1000 images
- Viral Pneumonia: 1000 images
- Training: 2100 images
- Validation: 450 images
- Testing: 450 images

## Epoch Results

| Epoch | Train Loss | Train Accuracy | Validation Loss | Validation Accuracy |
|---|---|---|---|---|
| 1 | 1.0179 | 50.90% | 0.9266 | 61.78% |
| 2 | 0.8823 | 67.43% | 0.8273 | 68.00% |
| 3 | 0.7892 | 73.95% | 0.7486 | 72.44% |
| 5 | 0.6643 | 78.76% | 0.6606 | 75.33% |

## Best Model
- File: models/best_resnet18.pth
- Best validation accuracy: 75.33%

## Notes
This is an initial baseline using a frozen ResNet-18 backbone.
Independent test evaluation, per-class metrics, ROC-AUC,
and further model improvements are pending.

This model is for educational research only and is not
intended for clinical diagnosis.
