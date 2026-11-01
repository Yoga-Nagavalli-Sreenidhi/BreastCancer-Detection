import streamlit as st
import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from math import pi
from mpl_toolkits.mplot3d import Axes3D

# Load the model and scaler
def load_model():
    with open('best_model.sav', 'rb') as file:
        loaded_model = pickle.load(file)
    model = loaded_model['model']  # Access the 'model' from the dictionary
    scaler = loaded_model['scaler']  # Access the 'scaler' from the dictionary
    metrics = loaded_model.get('metrics', None)  # Access metrics if saved
    return model, scaler, metrics

# Sidebar text input for features
def get_user_input():
    st.sidebar.title("Patient Feature Input")
    st.sidebar.write("Enter the values for the features below:")

    features = {}
    feature_names = ['texture_worst', 'compactness_se', 'concavity_worst', 'concave_points_mean', 'texture_mean', 'area_se', 'area_worst', 'perimeter_worst', 'radius_se', 'concave_points_worst', 'smoothness_worst', 'symmetry_worst', 'symmetry_se']
    
    for i, name in enumerate(feature_names, start=1):
        value = st.sidebar.text_input(f"{name}:", value="0.0")  # Default to 0.0
        try:
            features[f'feature_{i}'] = float(value)
        except ValueError:
            st.sidebar.warning(f"Please enter a valid number for {name}.")
            features[f'feature_{i}'] = 0.0

    input_data = np.array([list(features.values())])
    return input_data, features

# Display dataset information
def display_dataset_info():
    st.sidebar.title("Dataset Information")
    st.sidebar.write(""" 
    *Dataset*: Breast Cancer Wisconsin (Diagnostic) Dataset  
    *Source*: UCI Machine Learning Repository  
    *Total Records*: 569  
    *Features*: 30  
    *Classes*: Benign (Non-cancerous) and Malignant (Cancerous)  
    """)
    st.sidebar.markdown(""" 
    <style>
    .sidebar-info {
        background-color: #e0fbfc;
        padding: 10px;
        border-radius: 5px;
        font-family: Arial, sans-serif;
    }
    </style>
    """, unsafe_allow_html=True)

# Visualize dataset composition
def visualize_dataset_composition():
    st.markdown("### Dataset Composition")
    benign_count = 357  # Example count
    malignant_count = 212  # Example count

    labels = ["Benign", "Malignant"]
    sizes = [benign_count, malignant_count]
    colors = ['#007f5f', '#d00000']
    explode = (0.1, 0)  # Explode the first slice for emphasis

    fig, ax = plt.subplots()
    ax.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
           shadow=True, startangle=90)
    ax.axis('equal')  # Equal aspect ratio ensures a circular pie chart
    st.pyplot(fig)

# Medical-themed report display

def display_report(prediction, metrics):
    st.subheader("Prediction Report")

    result_class = str(prediction[0])

    # Map the model's original class labels
    if result_class in ["0", "B", "Benign"]:
        result_class = "Benign (Non-cancerous)"
        result_color = "green"
    elif result_class in ["1", "M", "Malignant"]:
        result_class = "Malignant (Cancerous)"
        result_color = "red"
    else:
        result_class = f"Model output: {result_class}"
        result_color = "black"

    st.markdown(
        f"""
        <div style="
            background-color: #ffffff;
            color: #222222;
            padding: 24px;
            border-radius: 12px;
            border: 2px solid #2196f3;
        ">
            <h3 style="color: {result_color};">
                Prediction: {result_class}
            </h3>
            <p><b>Model Accuracy:</b> {metrics['accuracy'] * 100:.2f}%</p>
            <p><b>Model Precision:</b> {metrics['precision'] * 100:.2f}%</p>
            <p><b>Model F1 Score:</b> {metrics['f1_score'] * 100:.2f}%</p>
        </div>
        """,
        unsafe_allow_html=True
    )



def main():
    model, scaler, metrics = load_model()

    st.markdown("""
    <style>
    .main-title {
        font-size: 2.2rem;
        color: #023e8a;
        text-align: center;
        font-weight: bold;
    }
    .sub-title {
        font-size: 1.1rem;
        color: #0077b6;
        text-align: center;
        margin-bottom: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown(
        '<h1 class="main-title">Breast Cancer Detection Using AdaBoost Classifier</h1>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<p class="sub-title">Machine learning project demonstration</p>',
        unsafe_allow_html=True
    )

    with st.expander("About Breast Cancer", expanded=False):
        st.write(
            "Breast cancer classification is demonstrated here using "
            "machine learning. This application is for educational use only."
        )
        st.image(
            "BCD.jpg",
            caption="Breast cancer awareness",
            use_container_width=True
        )

    with st.expander("About AdaBoost Classifier", expanded=False):
        st.write(
            "AdaBoost combines multiple weak learners to create a "
            "stronger classifier."
        )
        st.image(
            "model.jpg",
            caption="Project model illustration",
            use_container_width=True
        )

    display_dataset_info()
    visualize_dataset_composition()

    st.subheader("Enter Patient Feature Values")
    input_array, features = get_user_input()

    if st.button("Predict", key="predict_button"):
        try:
            scaled_input = scaler.transform(input_array)
            prediction = model.predict(scaled_input)
            display_report(prediction, metrics)
        except Exception as e:
            st.error(f"Prediction error: {e}")

    st.warning(
        "Educational demo only. This prediction is not a medical diagnosis."
    )


if __name__ == "__main__":
    main()
