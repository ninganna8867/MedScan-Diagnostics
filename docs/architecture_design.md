\# MedScan Diagnostics – Architecture Design



\## 1. Project Goal



Develop a chest X-ray image classification system using deep learning.



The selected dataset contains three classes:

\- COVID

\- Normal

\- Viral Pneumonia



\*\*Note:\*\* Viral Pneumonia is the dataset's actual class name. It is not a generic Pneumonia category.



\## 2. Model Architecture



\- Architecture: ResNet-18

\- Approach: Transfer Learning

\- Pretrained weights: ImageNet

\- Input image size: 224 × 224

\- Input channels: 3 (RGB)

\- Output classes: 3



\## 3. Classification Layer



\- Replace the original ResNet-18 final fully connected layer.

\- Use a new fully connected layer with 3 output neurons.

\- Each output represents one of the three dataset classes.



\## 4. Baseline Training Configuration



\- Loss function: CrossEntropyLoss

\- Optimizer: Adam

\- Learning rate: 0.0001

\- Batch size: 32

\- Epochs: 5

\- Training approach: Freeze the pretrained ResNet-18 backbone and train only the final fully connected layer.

\- Model selection: Save the model with the best validation accuracy.



\## 5. Dataset and Preprocessing



\- Total selected images: 3,000

\- Training set: 2,100 images

\- Validation set: 450 images

\- Testing set: 450 images

\- Image resizing: 224 × 224

\- Normalization: ImageNet mean and

