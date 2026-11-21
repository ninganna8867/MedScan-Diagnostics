

\# MedScan Diagnostics – Checkpoint 3

\## Core Model and Experimentation Report



\### 1. Model Architecture

\- Architecture: ResNet-18

\- Pretrained weights: ImageNet

\- Framework: PyTorch

\- Dataset: 3,000 chest X-ray images

\- Classes: COVID, Normal, Viral Pneumonia

\- Input size: 224 × 224

\- Loss function: CrossEntropyLoss

\- Optimizer: Adam



\### 2. Experiment Tracking

MLflow 3.18.0 was used to track two model configurations.



Experiment name: MedScan\_Hyperparameter\_Experiments



\### 3. Hyperparameter Configurations



| Parameter | Configuration 1 | Configuration 2 |

|---|---:|---:|

| Learning rate | 0.0001 | 0.00001 |

| Batch size | 32 | 32 |

| Epochs | 5 | 1 |

| Trainable layers | Final FC layer | Final FC layer |



Configuration 1 uses the existing baseline model.

Configuration 2 starts from the saved baseline checkpoint and trains the final classification layer for one additional epoch.



\### 4. Validation Results



| Metric | Configuration 1 | Configuration 2 |

|---|---:|---:|

| Validation Accuracy | 75.33% | 76.00% |

| Validation Loss | 0.6606 | 0.6541 |



\### 5. Independent Test Results



| Metric | Configuration 1 | Configuration 2 |

|---|---:|---:|

| Test Accuracy | 75.78% | 76.44% |

| Macro ROC-AUC | 0.9263 | 0.9271 |

| COVID Recall | 40% | 42% |

| COVID False Negative Rate | 60% | 58% |



\### 6. Configuration 2 Classification Report



| Class | Precision | Recall | F1-score |

|---|---:|---:|---:|

| COVID | 0.93 | 0.42 | 0.58 |

| Normal | 0.60 | 0.93 | 0.73 |

| Viral Pneumonia | 0.93 | 0.95 | 0.94 |



\### 7. Configuration 2 Confusion Matrix



| Actual / Predicted | COVID | Normal | Viral Pneumonia |

|---|---:|---:|---:|

| COVID | 63 | 83 | 4 |

| Normal | 5 | 139 | 6 |

| Viral Pneumonia | 0 | 8 | 142 |



\### 8. Best Configuration



Configuration 2 achieved the better measured results:

\- Test accuracy: 76.44%

\- Macro ROC-AUC: 0.9271

\- Validation accuracy: 76.00%



It improved test accuracy by 0.66 percentage points compared with the baseline.



\### 9. Limitations and Future Improvements



\- COVID false-negative rate remains 58%, above the project target of less than 15%.

\- Further class-specific optimization and evaluation are required.

\- The two configurations used different numbers of training epochs, so the comparison is preliminary.

\- The test set should remain reserved for final evaluation rather than repeated tuning.



\### 10. Conclusion



Two ResNet-18 configurations were evaluated, and experiment parameters and metrics were recorded using MLflow. Configuration 2 achieved slightly better validation and independent test results.



The model requires further improvement before meeting all project acceptance criteria.



Disclaimer: This project is for educational research only and is not intended for clinical diagnosis.



