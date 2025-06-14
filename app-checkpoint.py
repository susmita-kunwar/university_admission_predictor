# University Admission Prediction System
# Author: [Your Name]
# Date: [Current Date]
# Description: A Streamlit application for analyzing and predicting university admission chances

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, confusion_matrix, classification_report,
                           roc_curve, auc, roc_auc_score, ConfusionMatrixDisplay)

# Set page configuration for the Streamlit app
st.set_page_config(page_title="University Admission Analysis", layout="wide")

# Define paths for model and data files
BASE_DIR = r"C:\Users\HP\Desktop\Project _UAP\university_admission_predictor"
MODEL_PATH = "model_with_features.pkl"

# Load the trained model, scaler, and feature list
@st.cache_resource  # Cache to improve performance
def load_model_data():
    """
    Loads the pre-trained model, scaler, and feature list from a saved file.
    Returns None for each component if loading fails.
    """
    try:
        model_data = joblib.load(MODEL_PATH)
        model = model_data['model']
        scaler = model_data['scaler']
        features = model_data['features']
        return model, scaler, features
    except Exception as e:
        st.error(f"Error loading model data: {str(e)}")
        return None, None, None

# Load model components
model, scaler, features = load_model_data()

# Create sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "EDA", "Model Predictions", "Model Analysis"])

# Home Page
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

# Exploratory Data Analysis Page
elif page == "EDA":
    st.title("Exploratory Data Analysis")
    
    @st.cache_data  # Cache data loading for better performance
    def load_data():
        """
        Loads the university admission dataset from CSV file.
        Returns an empty DataFrame if loading fails.
        """
        try:
            return pd.read_csv("university_admission.csv")  
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
        
        # Interactive visualizations
        st.subheader("Interactive Data Exploration")
        
        # 1. Correlation Heatmap
        with st.expander("Correlation Analysis"):
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(df.corr(), annot=True, cmap='coolwarm', ax=ax)
            st.pyplot(fig)
        
        # 2. Feature Distribution
        with st.expander("Feature Distributions"):
            selected_col = st.selectbox("Select feature", df.columns)
            fig, ax = plt.subplots()
            sns.histplot(df[selected_col], kde=True, ax=ax)
            st.pyplot(fig)
        
        # 3. Scatter Plot with Filters
        with st.expander("Relationship Explorer"):
            col1, col2 = st.columns(2)
            with col1:
                x_axis = st.selectbox("X-axis", df.columns, index=0)
            with col2:
                y_axis = st.selectbox("Y-axis", df.columns, index=6)  # Default to Chance of Admit
            
            fig, ax = plt.subplots()
            sns.scatterplot(data=df, x=x_axis, y=y_axis, hue='Chance of Admit', ax=ax)
            st.pyplot(fig)
        
        # 4. Boxplots for categorical analysis
        with st.expander("Category Analysis"):
            cat_feature = st.selectbox("Select categorical feature", 
                                     ['University_Rating', 'Research', 'SOP'])
            fig, ax = plt.subplots()
            sns.boxplot(data=df, x=cat_feature, y='Chance of Admit', ax=ax)
            st.pyplot(fig)

# Model Prediction Page
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
                
            with col2:
                sop = st.slider("Statement of Purpose Strength (1-5)", 1, 5, 3)
                lor = st.slider("Letter of Recommendation Strength (1-5)", 1, 5, 3)
                cgpa = st.slider("CGPA (6.0-10.0)", 6.0, 10.0, 8.5)
            
            submitted = st.form_submit_button("Predict Admission Probability")
        
        if submitted:
            # Prepare input data for prediction
            input_data = pd.DataFrame({
                'GRE_Score': [gre_score],
                'TOEFL_Score': [toefl_score],
                'SOP': [sop],
                'LOR': [lor],
                'CGPA': [cgpa]
            })
            
            # Ensure input matches the features order used in training
            input_data = input_data[features]
            
            # Scale the features using the same scaler from training
            input_scaled = scaler.transform(input_data)
            
            # Predict probability (if model supports probability estimates)
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(input_scaled)[:, 1][0]
            else:
                proba = model.predict(input_scaled)[0]
            
            st.subheader("Prediction Results")
            
            # Visualize admission probability as a horizontal bar
            fig, ax = plt.subplots(figsize=(8, 1))
            ax.barh(['Admission Chance'], [1], color='lightgray')
            ax.barh(['Admission Chance'], [proba], color='#1f77b4')
            ax.set_xlim(0, 1)
            ax.set_title(f"Admission Probability: {proba:.1%}")
            ax.set_xticks([])
            st.pyplot(fig)
            
            # Provide interpretation based on probability
            if proba > 0.8:
                st.success("High Chance of Admission (Probability > 80%)")
            elif proba > 0.6:
                st.warning("Moderate Chance of Admission (60-80%)")
            else:
                st.error("Low Chance of Admission (Probability < 60%)")

# Model Analysis Page
elif page == "Model Analysis":
    st.title("Model Performance Analysis")
    
    # Enhanced model loading with fallback options
    @st.cache_resource
    def load_model_components():
        """
        Attempts to load model from local path first, then falls back to cloud sources.
        Returns None for each component if loading fails.
        """
        try:
            local_model_path = os.path.join(BASE_DIR, "model_with_features.pkl")
            
            if os.path.exists(local_model_path):
                model_data = joblib.load(local_model_path)
                return model_data['model'], model_data['scaler'], model_data['features']
            
            st.warning("Local model not found - checking cloud resources...")
            try:
                cloud_model_url = "https://github.com/susmita-kunwar/university_admission_predictor/raw/main/model_with_features.pkl"
                model_data = joblib.load(requests.get(cloud_model_url).content)
                return model_data['model'], model_data['scaler'], model_data['features']
            except:
                st.error("Could not load model from any source")
                return None, None, None
                
        except Exception as e:
            st.error(f"Model loading error: {str(e)}")
            return None, None, None

    model, scaler, features = load_model_components()
    
    if None in [model, scaler, features]:
        st.error("""
        Model components failed to load. Please:
        1. Check model_with_features.pkl exists in your project folder
        2. Verify the file contains 'model', 'scaler', and 'features'
        """)
        st.stop()

    # Data loading with multiple fallback options
    @st.cache_data
    def load_admission_data():
        """
        Attempts to load data from local path first, then falls back to cloud sources.
        Generates synthetic data if neither source is available.
        """
        try:
            local_data_path = os.path.join(BASE_DIR, "university_admission.csv")
            
            if os.path.exists(local_data_path):
                return pd.read_csv(local_data_path)
            
            st.warning("Local data not found - trying cloud sources...")
            try:
                github_url = "https://raw.githubusercontent.com/susmita-kunwar/university_admission_predictor/main/university_admission.csv"
                return pd.read_csv(github_url)
            except:
                st.warning("Using synthetic sample data")
                return pd.DataFrame({
                    'GRE_Score': np.random.randint(290, 340, 100),
                    'TOEFL_Score': np.random.randint(90, 120, 100),
                    'University_Rating': np.random.randint(1, 5, 100),
                    'SOP': np.random.uniform(1, 5, 100).round(1),
                    'LOR': np.random.uniform(1, 5, 100).round(1),
                    'CGPA': np.random.uniform(6.5, 9.9, 100).round(1),
                    'Research': np.random.randint(0, 2, 100),
                    'Chance of Admit': np.random.uniform(0.4, 0.95, 100).round(2)
                })
                
        except Exception as e:
            st.error(f"Data loading failed: {str(e)}")
            return pd.DataFrame()

    df = load_admission_data()
    
    if df.empty:
        st.error("No data available - cannot perform analysis")
        st.stop()

    # Check for missing features and handle them
    missing_features = [f for f in features if f not in df.columns]
    if missing_features:
        st.error(f"Missing required features: {', '.join(missing_features)}")
        st.warning(f"Available features: {', '.join(df.columns)}")
        
        # Generate dummy values for missing features
        for f in missing_features:
            if f == 'Research':
                df[f] = np.random.randint(0, 2, len(df))
            elif 'Score' in f or 'CGPA' in f:
                df[f] = np.random.normal(300, 50, len(df)).clip(260, 340)
            else:
                df[f] = 0
        st.warning(f"Generated dummy values for missing features")

    # Interactive model evaluation section
    st.subheader("Model Performance Metrics")
    
    # Threshold selection for classification
    threshold = st.slider(
        "Admission Chance Threshold", 
        min_value=0.1, 
        max_value=0.9, 
        value=0.75, 
        step=0.05,
        help="Adjust to change what counts as 'Admitted'"
    )
    
    # Prepare features and target variable
    X = df[features]
    y = (df['Chance of Admit'] > threshold).astype(int)
    
    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    try:
        # Scale test data and make predictions
        X_test_scaled = scaler.transform(X_test)
        y_pred = model.predict(X_test_scaled)
        y_proba = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, 'predict_proba') else y_pred
        
        # Calculate evaluation metrics
        metrics = {
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred),
            "Recall": recall_score(y_test, y_pred),
            "F1 Score": f1_score(y_test, y_pred),
            "ROC AUC": roc_auc_score(y_test, y_proba)
        }
        
        # Display metrics in a styled dataframe
        col1, col2 = st.columns(2)
        with col1:
            st.dataframe(
                pd.DataFrame.from_dict(metrics, orient='index', columns=['Value'])
                .style.format("{:.2%}")
                .background_gradient(cmap='Blues')
            )
        
        # Display model information
        with col2:
            st.metric("Model Type", type(model).__name__)
            st.metric("Feature Count", len(features))
            st.metric("Test Samples", len(X_test))
        
        # Create tabs for different visualizations
        tab1, tab2, tab3 = st.tabs(["Confusion Matrix", "Classification Report", "ROC Curve"])
        
        with tab1:
            fig, ax = plt.subplots()
            ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax, cmap='Blues')
            ax.set_title(f"Threshold: {threshold:.0%}")
            st.pyplot(fig)
        
        with tab2:
            report = classification_report(y_test, y_pred, output_dict=True)
            st.dataframe(
                pd.DataFrame(report).transpose()
                .style.format("{:.2%}", subset=['precision', 'recall', 'f1-score'])
                .format("{:.0f}", subset=['support'])
                .background_gradient(cmap='Blues')
            )
        
        with tab3:
            if hasattr(model, 'predict_proba'):
                fpr, tpr, _ = roc_curve(y_test, y_proba)
                roc_auc = auc(fpr, tpr)
                
                fig, ax = plt.subplots()
                ax.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
                ax.plot([0, 1], [0, 1], 'k--')
                ax.set_xlabel('False Positive Rate')
                ax.set_ylabel('True Positive Rate')
                ax.set_title('Receiver Operating Characteristic')
                ax.legend(loc="lower right")
                st.pyplot(fig)
            else:
                st.warning("This model doesn't support probability estimates")
                
    except Exception as e:
        st.error(f"Evaluation failed: {str(e)}")
        st.error("Possible model-data mismatch - check feature scaling")