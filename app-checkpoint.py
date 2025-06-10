import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, confusion_matrix, classification_report,
                           roc_curve, auc)

# Set page config
st.set_page_config(page_title="University Admission Analysis", layout="wide")

# Define paths using Path for cross-platform compatibility
BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "model_with_features.pkl"
DATA_PATH = BASE_DIR / "university_admission.csv"

# Load the model, scaler, and features with robust error handling
@st.cache_resource
def load_model_data():
    try:
        # Verify model file exists
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
        
        model_data = joblib.load(MODEL_PATH)
        
        # Validate loaded data structure
        required_keys = {'model', 'scaler', 'features'}
        if not all(key in model_data for key in required_keys):
            raise ValueError("Model data is missing required components")
            
        return model_data['model'], model_data['scaler'], model_data['features']
        
    except Exception as e:
        st.error(f"Error loading model data: {str(e)}")
        st.error(f"Expected model file at: {MODEL_PATH.absolute()}")
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
    - GRE Scores (260-340)
    - TOEFL Scores (90-120)
    - Academic Performance (CGPA 6.0-10.0)
    - Statement of Purpose Strength (1-5)
    - Letters of Recommendation Strength (1-5)
    
    ### Key Features:
    - Exploratory Data Analysis
    - Admission Probability Prediction
    - Model Performance Metrics
    
    Use the sidebar to navigate through different sections.
    """)

elif page == "EDA":
    st.title("Exploratory Data Analysis")
    
    @st.cache_data
    def load_data():
        try:
            if not DATA_PATH.exists():
                raise FileNotFoundError(f"Data file not found at {DATA_PATH}")
            return pd.read_csv(DATA_PATH)
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
            st.dataframe(df.describe().round(2))
        
        st.subheader("Data Visualizations")
        
        # Define numeric columns
        numeric_cols = ['GRE_Score', 'TOEFL_Score', 'University_Rating', 
                       'SOP', 'LOR', 'CGPA', 'Research', 'Chance of Admit']
        
        # Distribution plots
        st.markdown("### Feature Distributions")
        selected_dist = st.selectbox("Select feature to view distribution", numeric_cols)
        
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.histplot(data=df, x=selected_dist, kde=True, bins=20, 
                    color='skyblue', edgecolor='black')
        plt.title(f'Distribution of {selected_dist}')
        st.pyplot(fig)
        
        # Correlation analysis
        st.markdown("### Correlation Analysis")
        fig, ax = plt.subplots(figsize=(10, 8))
        corr = df[numeric_cols].corr().round(2)
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", 
                   center=0, vmin=-1, vmax=1, ax=ax)
        plt.title("Feature Correlation Matrix")
        st.pyplot(fig)
        
        # Feature vs Target
        st.markdown("### Feature vs Admission Chance")
        feature = st.selectbox("Select feature to compare with", 
                              ['GRE_Score', 'TOEFL_Score', 'CGPA', 'SOP', 'LOR'])
        
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.regplot(data=df, x=feature, y='Chance of Admit', 
                   scatter_kws={'alpha':0.3, 'color':'blue'},
                   line_kws={'color':'red'})
        plt.title(f'{feature} vs Admission Chance')
        st.pyplot(fig)

elif page == "Model Predictions":
    st.title("Admission Predictions")
    
    if model is None or scaler is None or features is None:
        st.error("Model components failed to load. Please verify:")
        st.error(f"- Model file exists at: {MODEL_PATH}")
        st.error("- File contains 'model', 'scaler', and 'features'")
    else:
        st.subheader("Predict Admission Probability")
        st.info(f"Model loaded: {type(model).__name__}")
        st.info(f"Features required (in order): {features}")
        
        with st.form("prediction_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                gre_score = st.slider("GRE Score (260-340)", 260, 340, 310)
                toefl_score = st.slider("TOEFL Score (90-120)", 90, 120, 105)
                
            with col2:
                cgpa = st.slider("CGPA (6.0-10.0)", 6.0, 10.0, 8.5)
                sop = st.slider("SOP Strength (1-5)", 1, 5, 3)
                lor = st.slider("LOR Strength (1-5)", 1, 5, 3)
            
            submitted = st.form_submit_button("Predict Admission Probability")
        
        if submitted:
            try:
                # Create input DataFrame with exact feature order
                input_data = pd.DataFrame([{
                    'GRE_Score': gre_score,
                    'TOEFL_Score': toefl_score,
                    'CGPA': cgpa,
                    'SOP': sop,
                    'LOR': lor
                }])[features]  # Ensure correct column order
                
                # Scale features
                input_scaled = scaler.transform(input_data)
                
                # Predict
                if hasattr(model, 'predict_proba'):
                    proba = model.predict_proba(input_scaled)[:, 1][0]
                else:
                    proba = model.predict(input_scaled)[0]
                
                # Display results
                st.subheader("Prediction Results")
                
                # Probability bar
                fig, ax = plt.subplots(figsize=(8, 1))
                ax.barh(['Probability'], [1], color='lightgray')
                ax.barh(['Probability'], [proba], color='#1f77b4')
                ax.set_xlim(0, 1)
                ax.set_title(f"Admission Probability: {proba:.1%}")
                ax.set_xticks([])
                st.pyplot(fig)
                
                # Interpretation
                if proba > 0.8:
                    st.success("High Chance of Admission (>80%)")
                elif proba > 0.6:
                    st.warning("Moderate Chance (60-80%)")
                else:
                    st.error("Low Chance (<60%)")
                    
            except Exception as e:
                st.error(f"Prediction failed: {str(e)}")

elif page == "Model Analysis":
    st.title("Model Performance Analysis")
    
    if model is None:
        st.error("Model not loaded. Please check the model file.")
    else:
        st.subheader("Model Diagnostics")
        
        # Load data
        try:
            df = pd.read_csv(DATA_PATH)
            X = df[features]
            y = (df['Chance of Admit'] > 0.75).astype(int)  # Binary target
            
            # Split data
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42)
            
            # Scale test data
            X_test_scaled = scaler.transform(X_test)
            
            # Predictions
            y_pred = model.predict(X_test_scaled)
            
            # Metrics
            st.subheader("Classification Report")
            report = classification_report(y_test, y_pred, output_dict=True)
            st.dataframe(pd.DataFrame(report).transpose().style.format("{:.2f}"))
            
            # Confusion Matrix
            st.subheader("Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=['Rejected', 'Admitted'],
                        yticklabels=['Rejected', 'Admitted'])
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            st.pyplot(fig)
            
            # ROC Curve (if classifier supports probabilities)
            if hasattr(model, 'predict_proba'):
                st.subheader("ROC Curve")
                y_proba = model.predict_proba(X_test_scaled)[:, 1]
                fpr, tpr, _ = roc_curve(y_test, y_proba)
                roc_auc = auc(fpr, tpr)
                
                fig, ax = plt.subplots()
                ax.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
                ax.plot([0, 1], [0, 1], 'k--')
                ax.set_xlabel('False Positive Rate')
                ax.set_ylabel('True Positive Rate')
                ax.set_title('ROC Curve')
                ax.legend()
                st.pyplot(fig)
                
        except Exception as e:
            st.error(f"Analysis failed: {str(e)}")