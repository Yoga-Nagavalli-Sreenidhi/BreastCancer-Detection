# Breast Cancer Detection Using AdaBoost Classifier

## Project Overview

This project demonstrates a machine learning-based approach to breast cancer classification through a Streamlit web application.

The application allows users to enter selected feature values and view a model prediction along with evaluation metrics.

## Objectives

- Demonstrate a machine learning classification workflow.
- Understand the AdaBoost classifier.
- Provide an interactive web interface for model predictions.
- Display model evaluation metrics.

## Technologies Used

- Python
- Streamlit
- NumPy
- Pandas
- Scikit-learn
- Imbalanced-learn

## Dataset Information

- **Dataset:** Breast Cancer Wisconsin (Diagnostic) Dataset
- **Total records:** 569
- **Features:** 30
- **Classes:** Benign and Malignant

## Application Features

- Patient feature input interface
- Classification prediction display
- Model evaluation metrics
- Dataset information and visualizations

## Installation

Install the required dependencies:

```bash
pip install -r BCD_app/requirements.txt
```

## Run the Application

From the repository root directory, run:

```bash
python -m streamlit run BCD_app/app.py
```

If the application file is in a different folder, adjust the path accordingly.

## Disclaimer

This application is intended for educational demonstration only. It is not a medical diagnostic tool.

## Acknowledgements

Project materials and code derived from existing work should be credited to their original contributors. Any additional modifications should be documented separately.

Developed an interactive web application using Streamlit, allowing users to input patient data and get real-time predictions.
