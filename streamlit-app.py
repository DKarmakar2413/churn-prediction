import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Streamlit App Title
st.title("Customer Churn Prediction")

# Sidebar for user input
st.sidebar.header("Upload Dataset")

# File Upload
uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])
if uploaded_file is not None:
    # Load dataset
    df = pd.read_csv(uploaded_file)
    st.write("Dataset Preview:")
    st.dataframe(df.head())

    # Preprocess the dataset
    st.header("Data Preprocessing")
    st.write("Removing unnecessary columns...")
    df.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1, inplace=True)

    # Encode categorical features
    st.write("Encoding categorical variables...")
    le = LabelEncoder()
    df['Geography'] = le.fit_transform(df['Geography'])
    df['Gender'] = le.fit_transform(df['Gender'])

    # Scale numerical features
    scaler = StandardScaler()
    numeric_cols = ['CreditScore', 'Balance', 'EstimatedSalary']
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    st.write("Processed Dataset:")
    st.dataframe(df.head())

    # Split data
    X = df.drop('Exited', axis=1)
    y = df['Exited']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a model
    st.header("Model Training")
    st.write("Training a Random Forest Classifier...")
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Show model performance
    st.write("Model Accuracy on Test Set:", model.score(X_test, y_test))

    # Feature Importance
    st.header("Feature Importance")
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': model.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    st.bar_chart(feature_importance.set_index('Feature'))

# Handle case when no file is uploaded
else:
    st.write("Awaiting file upload. Please upload a CSV file from the sidebar.")

