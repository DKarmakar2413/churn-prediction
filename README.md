# Customer Churn Prediction Project - README

## Project Overview
This project focuses on predicting customer churn for a subscription-based service using machine learning. By analyzing patterns in customer behavior, it helps identify customers who are likely to discontinue the service, enabling proactive retention strategies.

### Key Features:
- **Exploratory Data Analysis (EDA):** Understand customer trends through visualizations.
- **Data Preprocessing:** Handle missing values, encode categorical features, and scale numeric features for better model performance.
- **Feature Engineering:** Derive new insights from existing data to improve prediction accuracy.
- **Model Training and Evaluation:** Compare multiple machine learning models (Random Forest, XGBoost, Gradient Boosting) using metrics like ROC-AUC and classification reports.
- **Feature Importance Analysis:** Identify the most influential factors contributing to customer churn.

### Tools and Libraries:
- Python (pandas, numpy, matplotlib, seaborn, scikit-learn, XGBoost, imbalanced-learn)
- Streamlit for web-based visualization and interaction
- SMOTE for handling class imbalance

## How to Use the Project
1. Clone this repository and install the required dependencies from `requirements.txt`.
2. Run the Python script to train the models and generate predictions.
3. View the visualized results and feature importance in the Streamlit app.

### Streamlit App
Check out the interactive version of the project here:  
[Customer Churn Analysis - Streamlit App](https://churn-analysis-pred.streamlit.app/) 

This app allows you to explore the data, view the churn predictions, and interact with model results in real-time.
