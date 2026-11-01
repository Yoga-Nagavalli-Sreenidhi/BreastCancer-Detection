#Breast Cancer Detection System Using Machine Learning

This repository contains the implementation of a Breast Cancer Detection System developed as part of the Infosys Springboard Internship (October 3, 2024 – December 5, 2024). The project leverages machine learning techniques to accurately classify tumors as malignant or benign, enabling early detection and diagnosis.
Demo Link: https://breastcancer-detection04.streamlit.app/

🔍 Overview
Early detection of breast cancer significantly improves survival rates. This project aims to provide a robust solution using the AdaBoost Classifier to enhance prediction accuracy and reduce false negatives, which are critical in healthcare applications.

🛠️ Technologies Used
Programming Language: Python
Machine Learning Libraries: scikit-learn, imbalanced-learn
Data Manipulation: pandas, numpy
Visualization Tools: matplotlib, seaborn
Deployment: Streamlit
📊 Key Features
Data Preprocessing:

Handled missing values and encoded categorical data.
Scaled features using StandardScaler for optimal performance.
Class Imbalance Handling:

Addressed class imbalance using SMOTE (Synthetic Minority Over-sampling Technique).
Model Development:

Built a robust AdaBoost Classifier using decision trees as weak learners.
Improved performance through GridSearchCV for hyperparameter tuning.
Evaluation Metrics:

Evaluated using precision, recall, F1-score, and AUC-ROC for reliable diagnostics.
Feature Selection:

Identified key features with importance >3% and retrained the model for enhanced efficiency.
Deployment:

Developed an interactive web application using Streamlit, allowing users to input patient data and get real-time predictions.
