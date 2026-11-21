\# MedScan Diagnostics – Dataset Pipeline



\## 1. Dataset Classes

\- Normal

\- Pneumonia

\- COVID-19



\## 2. Dataset Structure

\- data/raw: Original X-ray images

\- data/processed: Prepared dataset



\## 3. Data Validation

\- Check image formats.

\- Check image readability.

\- Verify class labels.

\- Identify duplicate images.

\- Check missing or corrupted files.



\## 4. Dataset Splitting

\- Training: 70%

\- Validation: 15%

\- Testing: 15%



Patient-level splitting will be used when patient IDs

are available.



\## 5. Image Preprocessing

\- Resize images to 224 x 224.

\- Convert images to RGB.

\- Convert images into tensors.

\- Normalize using ImageNet mean and standard deviation.



\## 6. Data Augmentation

Training images may use:

\- Small rotations.

\- Horizontal flipping when medically appropriate.

\- Small brightness and contrast changes.



Validation and test images will not use random augmentation.



\## 7. Framework

\- Python

\- Pillow

\- NumPy

\- PyTorch

\- Torchvision



\## 8. Important Note

The actual dataset source and usage permissions must be

verified before downloading images.

