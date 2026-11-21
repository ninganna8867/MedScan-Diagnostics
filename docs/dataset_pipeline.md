\# MedScan Diagnostics – Dataset Pipeline



\## 1. Dataset Source



\- Dataset: COVID-19 Radiography Database

\- Source: https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database

\- Downloaded dataset size: Approximately 816 MB

\- Original dataset contains COVID, Lung\_Opacity, Normal and Viral Pneumonia classes.

\- This project uses only the images from the COVID, Normal and Viral Pneumonia folders.



\*\*License and usage:\*\* The Kaggle dataset page identifies the data files as belonging to their original authors and specifies academic/non-commercial use. Review the current dataset terms before redistribution or other use.



\## 2. Selected Dataset Classes



The project uses three classes:

\- COVID: 1,000 images

\- Normal: 1,000 images

\- Viral Pneumonia: 1,000 images



\*\*Total selected images: 3,000\*\*



Note: Viral Pneumonia is the actual dataset label. It is not a generic Pneumonia category.



\## 3. Dataset Structure



\- `data/raw/`: Original downloaded X-ray dataset.

\- `data/processed/`: Selected and split images.



The processed dataset is organized into:

\- `train/`

\- `val/`

\- `test/`



Each split contains the three class folders.



\## 4. Data Validation



The pipeline should:

\- Check image formats.

\- Check image readability.

\- Verify class labels.

\- Identify duplicate images.

\- Check missing or corrupted files.



\## 5. Dataset Splitting



The selected 3,000 images are split using a fixed random seed.



\- Training: 70% (2,100 images)

\- Validation: 15% (450 images)

\- Testing: 15% (450 images)



Each class contains:

\- Training: 700 images

\- Validation: 150 images

\- Testing: 150 images



\*\*Limitation:\*\* The current implementation performs image-level random splitting. Patient-level splitting has not been implemented. Possible patient overlap and data leakage must be considered when interpreting evaluation results.



\## 6. Image Preprocessing



\- Resize images to 224 × 224.

\- Convert images to RGB.

\- Convert images into PyTorch tensors.

\- Normalize using ImageNet mean and standard deviation.



\## 7. Data Augmentation



Training images currently use:

\- Random horizontal flipping.

\- Random rotation up to 10 degrees.



Validation and test images use deterministic preprocessing without random augmentation.



Medical appropriateness of augmentation choices should be reviewed.



\## 8. Framework and Libraries



\- Python

\- Pillow

\- NumPy

\- PyTorch

\- Torchvision



\## 9. Important Limitations



\- The selected dataset is not the same as the dataset originally described as 3,000 images in the project brief; it is an external public dataset used because the supplied data packs did not contain chest X-ray images.

\- The selected classes differ from the generic Normal, Pneumonia and COVID-19 labels in the brief.

\- Patient-level splitting has not been performed.

\- Dataset images are excluded from the GitHub repository.

\- The model is for educational research only and is not intended for medical diagnosis.



\## 10. Dataset Citation



M. E. H. Chowdhury et al., "Can AI help in screening Viral and COVID-19 pneumonia?" IEEE Access, Volume 8, 2020, pages 132665–132676.



Dataset source: https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database

