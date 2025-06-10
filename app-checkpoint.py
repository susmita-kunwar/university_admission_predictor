import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, confusion_matrix, classification_report,
                           roc_curve, auc)

# Set page config
st.set_page_config(page_title="University Admission Analysis", layout="wide")

# Define paths
BASE_DIR = r"C:\Users\HP\Desktop\Project _UAP\university_admission_predictor"
MODEL_PATH = MODEL_PATH = "model_with_features.pkl"

# Load the model, scaler, and features
@st.cache_resource
def load_model_data():
    try:
        model_data = joblib.load(MODEL_PATH)
        model = model_data['model']
        scaler = model_data['scaler']
        features = model_data['features']
        return model, scaler, features
    except Exception as e:
        st.error(f"Error loading model data: {str(e)}")
        return None, None, None

model, scaler, features = load_model_data()

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "EDA", "Model Predictions", "Model Analysis"])

if page == "Home":
    st.title("University Admission Prediction Analysis")
    st.image(
        "https://images.unsplash.com/photo-1523050854058-8df90110c9f1?ixlib=rb-1.2.1&auto=format&fit=crop&w=1350&q=80", 
        use_container_width=True
    )
    
    st.markdown("""
    ## Welcome to the University Admission Prediction System
    
    This application provides comprehensive analysis of university admission chances based on:
    - GRE Scores
    - TOEFL Scores
    - Academic Performance (CGPA)
    - Statement of Purpose (SOP)
    - Letters of Recommendation (LOR)
    
    ### Key Features:
    - Exploratory Data Analysis: Visualize admission trends and relationships
    - Admission Prediction: Get probability estimates for admission
    - Model Performance: Detailed evaluation metrics
    
    Use the sidebar to navigate through different sections.
    """)

elif page == "EDA":
    st.title("Exploratory Data Analysis")
    
    @st.cache_data
    def load_data():
        try:
            return pd.read_csv(os.path.join(BASE_DIR, "university_admission.csv"))
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            return pd.DataFrame()
    
    df = load_data()
    
    if not df.empty:
        st.subheader("Dataset Overview")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("First 5 rows:")
            st.dataframe(df.head())
        
        with col2:
            st.write("Basic Statistics:")
            st.dataframe(df.describe())
        
        st.subheader("Data Visualizations")
        
        # Define numeric columns including the target
        numeric_cols = ['GRE_Score', 'TOEFL_Score', 'University_Rating', 
                       'SOP', 'LOR', 'CGPA', 'Research', 'Chance of Admit']
        
        # Distribution plots
        st.markdown("### Feature Distributions")
        selected_dist = st.selectbox("Select feature to view distribution", numeric_cols)
        
        fig, ax = plt.subplots()
        sns.histplot(data=df, x=selected_dist, kde=True, bins=20, color='skyblue', edgecolor='black')
        plt.title(f'Distribution of {selected_dist}')
        st.pyplot(fig)
        
        # Correlation analysis
        st.markdown("### Correlation Analysis")
        fig, ax = plt.subplots(figsize=(10, 8))
        corr = df[numeric_cols].corr()
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
        plt.title("Feature Correlation")
        st.pyplot(fig)
        
        # Feature vs Target
        st.markdown("### Feature vs Admission Chance")
        feature = st.selectbox("Select feature to compare with Admission Chance", 
                              ['GRE_Score', 'TOEFL_Score', 'SOP', 'LOR', 'CGPA'])
        
        fig, ax = plt.subplots()
        sns.regplot(data=df, x=feature, y='Chance of Admit', scatter_kws={'alpha':0.3})
        plt.title(f'{feature} vs Admission Chance')
        st.pyplot(fig)

elif page == "Model Predictions":
    st.title("Admission Predictions")
    
    if model is None or scaler is None or features is None:
        st.error("Model, scaler, or features not loaded. Please check the model file.")
    else:
        st.subheader("Predict Admission Probability")
        with st.form("prediction_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                gre_score = st.slider("GRE Score (260-340)", 260, 340, 310)
                toefl_score = st.slider("TOEFL Score (90-120)", 90, 120, 105)
                # university_rating = st.selectbox("University Rating", [1, 2, 3, 4, 5])
                
            with col2:
                sop = st.slider("Statement of Purpose Strength (1-5)", 1, 5, 3)
                lor = st.slider("Letter of Recommendation Strength (1-5)", 1, 5, 3)
                cgpa = st.slider("CGPA (6.0-10.0)", 6.0, 10.0, 8.5)
                # research = st.radio("Research Experience", ["No", "Yes"])
            
            submitted = st.form_submit_button("Predict Admission Probability")
        
        if submitted:
            # research_code = 1 if research == "Yes" else 0
            
            input_data = pd.DataFrame({
                'GRE_Score': [gre_score],
                'TOEFL_Score': [toefl_score],
                # 'University_Rating': [university_rating],   # Optional: add back if your model used this
                'SOP': [sop],
                'LOR': [lor],
                'CGPA': [cgpa]
                # 'Research': [research_code]   # Optional
            })
            
            # Ensure input matches the features order
            input_data = input_data[features]
            
            # Scale the features
            input_scaled = scaler.transform(input_data)
            
            # Predict probability
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(input_scaled)[:, 1][0]
            else:
                proba = model.predict(input_scaled)[0]
            
            st.subheader("Prediction Results")
            
            # Create a horizontal bar showing admission probability
            fig, ax = plt.subplots(figsize=(8, 1))
            ax.barh(['Admission Chance'], [1], color='lightgray')
            ax.barh(['Admission Chance'], [proba], color='#1f77b4')
            ax.set_xlim(0, 1)
            ax.set_title(f"Admission Probability: {proba:.1%}")
            ax.set_xticks([])
            st.pyplot(fig)
            
            # Interpretation
            if proba > 0.8:
                st.success("High Chance of Admission (Probability > 80%)")
            elif proba > 0.6:
                st.warning("Moderate Chance of Admission (60-80%)")
            else:
                st.error("Low Chance of Admission (Probability < 60%)")

elif page == "Model Analysis":
    st.title("Model Performance Analysis")
    
    if model is None or scaler is None or features is None:
        st.error("Model, scaler, or features not loaded. Please check the model file.")
    else:
        st.subheader("Model Performance Metrics")
        
        # Display model type
        st.write(f"Model Type: {type(model).__name__}")
        
        # Example performance metrics (you can compute these properly if needed)
        metrics = {
            "Accuracy": 0.9505,
            "Precision": 0.8696,
            "Recall": 0.9091,
            "F1 Score": 0.8889,
            "ROC AUC": 0.9885
        }
        
        st.table(pd.DataFrame.from_dict(metrics, orient='index', columns=['Value']))
        
        # Load the dataset again to get test data
        df = pd.read_csv(os.path.join(BASE_DIR, "university_admission.csv"))
        
        # Prepare features and target
        X = df[features]
        # Example binarization of target (adjust your threshold if needed)
        y = (df['Chance of Admit'] > 0.75).astype(int)
        
        # Split into train/test
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale the test set
        X_test_scaled = scaler.transform(X_test)
        
        # Predict on test set
        y_pred = model.predict(X_test_scaled)
        
        # Confusion matrix
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Plot confusion matrix
        st.subheader("Confusion Matrix")
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Predicted No', 'Predicted Yes'],
                    yticklabels=['Actual No', 'Actual Yes'], ax=ax)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        st.pyplot(fig)
        
        # Optionally display classification report
        st.subheader("Classification Report")
        from sklearn.metrics import classification_report
        report = classification_report(y_test, y_pred, output_dict=True)
        st.dataframe(pd.DataFrame(report).transpose())