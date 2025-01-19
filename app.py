import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import kagglehub
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Configure the Streamlit page
st.set_page_config(page_title="Credit Card Fraud Detection", layout="wide")

# Sidebar Header
st.sidebar.title("⚙️ Controls")

# Download dataset
st.sidebar.write("📥 Downloading dataset...")
path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")
dataset_path = f"{path}/creditcard.csv"

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv(dataset_path)
    return df

df = load_data()

st.title("📊 Credit Card Fraud Detection Dashboard")

# Dataset Overview
st.write("## 📌 Dataset Overview")
col1, col2, col3 = st.columns(3)
col1.metric("Total Transactions", df.shape[0])
col2.metric("Fraudulent Transactions", df[df['Class'] == 1].shape[0])
col3.metric("Fraudulent Percentage", f"{100 * df['Class'].mean():.4f}%")

# Show dataset sample
if st.sidebar.checkbox("Show Sample Data"):
    st.write("### Raw Data Sample:")
    st.dataframe(df.head())

# Fraud vs Non-Fraud Bar Chart
st.write("## 🔍 Fraud vs Non-Fraud Transactions")
fig, ax = plt.subplots()
sns.countplot(data=df, x="Class", ax=ax)
ax.set_title("Fraud vs Non-Fraud Transactions")
ax.set_xticklabels(["Non-Fraud (0)", "Fraud (1)"])
st.pyplot(fig)

# Correlation Heatmap
if st.sidebar.checkbox("Show Correlation Heatmap"):
    st.write("## 🔥 Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(df.corr(), cmap="coolwarm", ax=ax)
    st.pyplot(fig)

# Machine Learning Model Training
st.write("## 🚀 Fraud Detection Model Training")

# Feature Selection
X = df.drop(columns=["Class", "Time"])
y = df["Class"]

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# Train Logistic Regression Model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Display model accuracy
st.write(f"### 🎯 Model Accuracy: **{accuracy:.4f}**")

# Show Confusion Matrix
if st.sidebar.checkbox("Show Confusion Matrix"):
    st.write("### 🟢 Confusion Matrix")
    fig, ax = plt.subplots()
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues", ax=ax)
    st.pyplot(fig)

# Classification Report
if st.sidebar.checkbox("Show Classification Report"):
    st.write("### 📋 Classification Report")
    st.text(classification_report(y_test, y_pred))

# Real-Time Fraud Detection Input
st.write("## 🔎 Predict Fraud for a Transaction")

# User Input Form
input_features = {}
for col in df.drop(columns=["Class", "Time"]).columns:
    input_features[col] = st.number_input(f"{col}", value=0.0)

# Predict Button
if st.button("Predict Fraud Risk"):
    input_df = pd.DataFrame([input_features])
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)
    probability = model.predict_proba(input_scaled)[0][1]

    if prediction[0] == 1:
        st.error(f"⚠️ High Fraud Risk! (Probability: {probability:.4f})")
    else:
        st.success(f"✅ Safe Transaction (Probability: {probability:.4f})")
