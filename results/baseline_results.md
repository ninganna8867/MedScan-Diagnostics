# MedScan - Baseline Training Results

## Model

* Architecture: ResNet-18
* Approach: Transfer Learning
* Pre-trained weights: ImageNet
* Trainable layers: Final fully connected layer
* Optimizer: Adam
* Learning rate: 0.0001
* Loss function: CrossEntropyLoss
* Epochs: 5
* Batch size: 32

## Dataset

* COVID: 1000 images
* Normal: 1000 images
* Viral Pneumonia: 1000 images
* Training: 2100 images
* Validation: 450 images
* Testing: 450 images

## Epoch Results

|Epoch|Train Loss|Train Accuracy|Validation Loss|Validation Accuracy|
|-|-|-|-|-|
|1|1.0179|50.90%|0.9266|61.78%|
|2|0.8823|67.43%|0.8273|68.00%|
|3|0.7892|73.95%|0.7486|72.44%|
|5|0.6643|78.76%|0.6606|75.33%|

## Best Model

* File: models/best\_resnet18.pth
* Best validation accuracy: 75.33%

## Notes

This is an initial baseline using a frozen ResNet-18 backbone.
Independent test evaluation, per-class metrics, ROC-AUC,
and further model improvements are pending.

This model is for educational research only and is not
intended for clinical diagnosis.

\## Independent Test Evaluation



\### Overall Results

\- Test Accuracy: 75.78%

\- Macro ROC-AUC (One-vs-Rest): 0.9263

\- Test Images: 450



\### Class-wise Results



| Class | Precision | Recall | F1-score |

|---|---:|---:|---:|

| COVID | 0.92 | 0.40 | 0.56 |

| Normal | 0.60 | 0.93 | 0.73 |

| Viral Pneumonia | 0.93 | 0.95 | 0.94 |



\### Confusion Matrix



| Actual / Predicted | COVID | Normal | Viral Pneumonia |

|---|---:|---:|---:|

| COVID | 60 | 86 | 4 |

| Normal | 5 | 139 | 6 |

| Viral Pneumonia | 0 | 8 | 142 |



\### COVID-19 False Negative Rate

\- COVID images: 150

\- Correct COVID predictions: 60

\- False negatives: 90

\- False Negative Rate: 60%



\### Observations

\- The model achieved 75.78% independent test accuracy.

\- Macro ROC-AUC was 0.9263.

\- COVID recall was only 40%, indicating that many COVID images were incorrectly classified as Normal.

\- Further model improvement and threshold analysis are required.

\- The project target of less than 15% COVID false-negative rate has not yet been achieved.



\*\*Conclusion:\*\* This is the initial baseline benchmark. The model requires further improvement before it can meet the project acceptance criteria.



\*\*Disclaimer:\*\* Educational research only; not for clinical diagnosis.

