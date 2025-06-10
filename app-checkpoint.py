import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import sys
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, confusion_matrix, classification_report,
                           roc_curve, auc)

# Set page config
st.set_page_config(page_title="University Admission Analysis", layout="wide")

# =============================================
# PATH HANDLING WITH ROBUST ERROR CHECKING
# =============================================
def setup_paths():
    """Handle file paths with comprehensive error checking"""
    try:
        # Get the directory where the script is running
        if getattr(sys, 'frozen', False):
            BASE_DIR = Path(sys.executable).parent
        else:
            BASE_DIR = Path(__file__).parent

        # Define file names
        MODEL_FILENAME = "model_with_features.pkl"
        DATA_FILENAME = "university_admission.csv"

        # Create paths
        MODEL_PATH = BASE_DIR / MODEL_FILENAME
        DATA_PATH = BASE_DIR / DATA_FILENAME

        # Debug output
        st.sidebar.markdown("### Path Diagnostics")
        st.sidebar.write(f"Script running from: {BASE_DIR}")
        st.sidebar.write(f"Model path: {MODEL_PATH}")
        st.sidebar.write(f"Data path: {DATA_PATH}")

        # Verify files exist
        if not MODEL_PATH.exists():
            available_files = "\n".join(os.listdir(BASE_DIR))
            raise FileNotFoundError(
                f"Model file not found at {MODEL_PATH}\n"
                f"Files in directory:\n{available_files}"
            )

        if not DATA_PATH.exists():
            raise FileNotFoundError(f"Data file not found at {DATA_PATH}")

        return BASE_DIR, MODEL_PATH, DATA_PATH

    except Exception as e:
        st.error(f"Path setup failed: {str(e)}")
        st.stop()

BASE_DIR, MODEL_PATH, DATA_PATH = setup_paths()

# =============================================
# MODEL LOADING WITH VALIDATION
# =============================================
@st.cache_resource
def load_model_data():
    try:
        model_data = joblib.load(MODEL_PATH)
        
        # Validate loaded data structure
        required_keys = {'model', 'scaler', 'features'}
        if not all(key in model_data for key in required_keys):
            raise ValueError(
                f"Model data missing required components\n"
                f"Expected: {required_keys}\n"
                f"Found: {list(model_data.keys())}"
            )
            
        return model_data['model'], model_data['scaler'], model_data['features']
        
    except Exception as e:
        st.error(f"Error loading model data: {str(e)}")
        st.error(f"Tried to load from: {MODEL_PATH}")
        st.error("Please verify:")
        st.error("1. The file exists at this location")
        st.error("2. It's a valid pickle file created with joblib")
        st.error("3. It contains 'model', 'scaler', and 'features'")
        st.stop()

model, scaler, features = load_model_data()

# =============================================
# DATA LOADING FUNCTION
# =============================================
@st.cache_data
def load_data():
    try:
        df = pd.read_csv(DATA_PATH)
        
        # Validate required columns exist
        required_cols = features + ['Chance of Admit']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            raise ValueError(
                f"Data missing required columns: {missing_cols}\n"
                f"Available columns: {list(df.columns)}"
            )
            
        return df
    
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        st.error(f"Tried to load from: {DATA_PATH}")
        st.stop()

# =============================================
# APPLICATION PAGES
# =============================================
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
    
    st.subheader("Predict Admission Probability")
    st.info(f"Model loaded: {type(model).__name__}")
    st.info(f"Features required (in order): {features}")
    
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            gre_score = st.slider("GRE Score (260-340)", 260, 340, 310)
            toefl_score = st.slider("TOEFL Score (90-120)", 90, 120, 105)
            
        with col2:
            cgpa = st.slider("CGPA (6.0-10.0)", 6.0, 10.0, 8.5, step=0.1)
            sop = st.slider("SOP Strength (1-5)", 1, 5, 3)
            lor = st.slider("LOR Strength (1-5)", 1, 5, 3)
        
        submitted = st.form_submit_button("Predict Admission Probability")
    
    if submitted:
        try:
            # Create input array maintaining exact feature order
            input_array = np.array([[gre_score, toefl_score, sop, lor, cgpa]])
            
            # Convert to DataFrame with correct column names
            input_df = pd.DataFrame(input_array, columns=features)
            
            # Scale features
            input_scaled = scaler.transform(input_df)
            
            # Predict
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(input_scaled)[0, 1]
            else:
                proba = model.predict(input_scaled)[0]
                proba = max(0, min(1, proba))  # Clamp between 0 and 1
            
            # Display results
            st.subheader("Prediction Results")
            
            # Probability bar with color gradient
            fig, ax = plt.subplots(figsize=(10, 1))
            cmap = plt.get_cmap('RdYlGn')
            rgba = cmap(proba)
            ax.barh(['Probability'], [1], color='lightgray')
            ax.barh(['Probability'], [proba], color=rgba)
            ax.set_xlim(0, 1)
            ax.set_title(f"Admission Probability: {proba:.1%}")
            ax.set_xticks([])
            st.pyplot(fig)
            
            # Interpretation
            if proba > 0.8:
                st.success("🎉 Excellent chance of admission (>80%)")
            elif proba > 0.6:
                st.warning("👍 Good chance of admission (60-80%)")
            else:
                st.error("⚠️ Lower chance of admission (<60%)")
                
        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")

elif page == "Model Analysis":
    st.title("Model Performance Analysis")
    
    df = load_data()
    X = df[features]
    y = (df['Chance of Admit'] > 0.75).astype(int)
    
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
    
    # Feature Importance (if available)
    if hasattr(model, 'feature_importances_'):
        st.subheader("Feature Importance")
        importance = pd.DataFrame({
            'Feature': features,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.barplot(data=importance, x='Importance', y='Feature', palette='viridis')
        st.pyplot(fig)
    
    # ROC Curve (if classifier supports probabilities)
    if hasattr(model, 'predict_proba'):
        st.subheader("ROC Curve")
        y_proba = model.predict_proba(X_test_scaled)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        
        fig, ax = plt.subplots()
        ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.2f}')
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.legend(loc="lower right")
        st.pyplot(fig)

