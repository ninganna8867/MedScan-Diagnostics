\# MedScan Diagnostics – Research Notes



\## Reference 1: ResNet Transfer Learning



\*\*Paper:\*\* Efficacy of Transfer Learning-based ResNet models in Chest X-ray image classification for detecting COVID-19 Pneumonia



\*\*Authors:\*\* Sadia Showkat and Shaima Qureshi  

\*\*Year:\*\* 2022  

\*\*Journal:\*\* Chemometrics and Intelligent Laboratory Systems  

\*\*DOI:\*\* https://doi.org/10.1016/j.chemolab.2022.104534  

\*\*Paper:\*\* https://pmc.ncbi.nlm.nih.gov/articles/PMC8913041/



\*\*Key findings:\*\*

\- Compares ResNet-18, ResNet-34, ResNet-50, ResNet-101 and ResNet-152.

\- Uses ImageNet-pretrained models and transfer learning.

\- Demonstrates the usefulness of ResNet architectures for chest X-ray classification.

\- Supports investigating ResNet-18 as our baseline model.



\## Reference 2: COVID-19 Classification and Grad-CAM



\*\*Paper:\*\* COVID-19 detection and disease progression visualization: Deep learning on chest X-rays for classification and coarse localization



\*\*Authors:\*\* Tahmina Zebin and Shahadate Rezvy  

\*\*Published online:\*\* 2020  

\*\*Journal volume:\*\* Applied Intelligence, Volume 51 (2021)  

\*\*DOI:\*\* https://doi.org/10.1007/s10489-020-01867-1  

\*\*Paper:\*\* https://link.springer.com/article/10.1007/s10489-020-01867-1



\*\*Key findings:\*\*

\- Uses transfer learning for chest X-ray classification.

\- Studies COVID-19, Pneumonia and Normal classes.

\- Implements Grad-CAM to highlight image regions influencing model predictions.

\- Supports adding explainability to our project.



\## Reference 3: Leakage-Aware Chest X-ray Classification



\*\*Repository:\*\* Reproducible Chest X-ray Pneumonia Classification Study  

\*\*GitHub:\*\* https://github.com/Evelyn-R7/chest-xray-pneumonia-classification



\*\*Key findings:\*\*

\- Emphasizes patient-level data splitting.

\- Discusses data leakage and model generalization.

\- Highlights the importance of reliable testing and reproducible evaluation.

\- Supports careful dataset preparation.



\*\*Note:\*\* This is a research and educational GitHub repository, not a peer-reviewed journal paper.



\## Research Conclusion



Based on the reviewed research, our project will begin with ResNet-18 transfer learning using ImageNet-pretrained weights.



Grad-CAM will be used to generate visual explanations of model predictions.



The model will be evaluated using:

\- Accuracy

\- Precision and Recall

\- Sensitivity and Specificity

\- ROC-AUC

\- COVID-19 false-negative rate



We will also document dataset limitations, possible data leakage, and model performance limitations.



\*\*Dataset note:\*\* Our selected dataset contains COVID, Normal and Viral Pneumonia classes. Viral Pneumonia is not identical to the broader Pneumonia category mentioned in some research papers.



\*\*Disclaimer:\*\* This project is for educational research only. It is not intended for clinical diagnosis or medical decision-making.

