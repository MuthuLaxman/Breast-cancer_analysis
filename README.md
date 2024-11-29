# Breast Cancer Prtion App

## Project Overview
This project is a web application built using Streamlit that predicts whether a breast tumor is malignant or benign. The model was trained on the Breast Cancer dataset from scikit-learn and is deployed as a Streamlit web app.

## Features
- Users can input values for the features of the dataset.
- The app uses a trained Random Forest model to predict if the tumor is malignant or benign.
- The app displays the prediction and confidence score.

## Files Included
- `train_model.py`: Script for training the model and saving the trained model and scaler.
- `app.py`: Streamlit app script that loads the trained model and provides predictions.
- `breast_cancer_model.pkl`: Trained machine learning model.
- `scaler.pkl`: Scaler used for normalizing input features.
- `README.md`: Project documentation.

## Instructions to Run the App
1. Install the required dependencies:
'''
   pip install pandas numpy scikit-learn streamlit joblib
'''

To train the model and save it, run:
'''
python train_model.py
'''

To run the Streamlit app:

'''
streamlit run app.py
'''

Open the Streamlit app in your browser at http://localhost:8502 and input the feature values to make predictions.edic
