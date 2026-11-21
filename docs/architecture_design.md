\# MedScan Diagnostics – Architecture Design



\## 1. Project Goal

Classify chest X-ray images into:

\- Normal

\- Pneumonia

\- COVID-19



\## 2. Model Architecture

\- Architecture: ResNet-18

\- Approach: Transfer Learning

\- Pretrained weights: ImageNet

\- Input image size: 224 x 224

\- Input channels: 3

\- Output classes: 3



\## 3. Classification Layer

Replace the original ResNet-18 final fully connected

layer with a new layer containing 3 output neurons.



\## 4. Training Configuration

\- Loss Function: CrossEntropyLoss

\- Optimizer: Adam

\- Evaluation: Validation and independent test sets



\## 5. Evaluation Metrics

\- Accuracy

\- Precision

\- Recall / Sensitivity

\- Specificity

\- ROC-AUC

\- COVID-19 False Negative Rate



\## 6. Explainability

\- Technique: Grad-CAM

\- Target layer: layer4\[-1]

\- Output: Heatmap over the input X-ray image



\## 7. Application

\- Framework: Gradio

\- Input: Uploaded chest X-ray

\- Output: Predicted class, class probabilities and Grad-CAM heatmap



\## 8. Important Considerations

\- Avoid patient-level data leakage wherever patient IDs are available.

\- Evaluate COVID-19 false negatives separately.

\- The project is educational and not intended for medical diagnosis.

