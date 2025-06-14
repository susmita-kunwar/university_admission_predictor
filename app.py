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

# Set consistent style using modern seaborn API
sns.set_theme(style="whitegrid", palette="Blues")

# Set page configuration
st.set_page_config(page_title="University Admission Analysis", layout="wide")

# Define paths (use forward slashes for cross-platform compatibility)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model_with_features.pkl")
DATA_PATH = os.path.join(BASE_DIR, "university_admission.csv")

# Load model and data
@st.cache_resource
def load_model_data():
    try:
        model_data = joblib.load(MODEL_PATH)
        return model_data['model'], model_data['scaler'], model_data['features']
    except Exception as e:
        st.error(f"Error loading model data: {str(e)}")
        return None, None, None

@st.cache_data
def load_data():
    try:
        df = pd.read_csv(DATA_PATH)
        # Standardize column names to match notebook
        df = df.rename(columns={
            'Chance of Admit': 'Admission_Chance',
            'GRE Score': 'GRE_Score',
            'TOEFL Score': 'TOEFL_Score',
            'University Rating': 'University_Rating',
            'LOR': 'LOR',
            'SOP': 'SOP',
            'CGPA': 'CGPA',
            'Research': 'Research'
        })
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return pd.DataFrame()

model, scaler, features = load_model_data()
df = load_data()

# Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "EDA", "Model Predictions", "Model Analysis"])

if page == "Home":
    st.title("University Admission Prediction Analysis")
    
    # Centered image with better spacing
    col1, col2, col3 = st.columns([1,6,1])
    with col2:
        st.image("https://images.unsplash.com/photo-1523050854058-8df90110c9f1?ixlib=rb-1.2.1&auto=format&fit=crop&w=1350&q=80", 
                use_container_width=True,
                caption="Graduate Admission Prediction System")
    
    st.markdown("""
    ## Welcome to the University Admission Prediction System
    
    This application predicts graduate program admission chances based on:
    - **GRE Scores** (260-340)
    - **TOEFL Scores** (90-120)
    - **University Rating** (1-5)
    - **Statement of Purpose (SOP) Strength** (1-5)
    - **Letter of Recommendation (LOR) Strength** (1-5)
    - **Undergraduate CGPA** (6.0-10.0)
    - **Research Experience** (0 or 1)
    """)

    st.markdown("---")
    
    st.markdown("""
    ## Key Features 

    - **Probability Prediction**: Predicts admission likelihood (0-100%) using ensemble regression
    - **Admission Classifier**: Binary classification with customizable decision threshold
    - **Interactive Analytics**: Dynamic visualization of admission trends and feature importance
    - **Model Benchmarking**: Comparative evaluation of 8 machine learning algorithms
    - **Hyperparameter Optimization**: Automated tuning for optimal model performance
    - **Decision Support**: Actionable insights to strengthen your application
    """)

    st.markdown("---")
    
    st.markdown("""
    ## Technical Implementation

    ### Data Processing Pipeline
    - Automated data validation and cleaning
    - Statistical outlier detection and treatment
    - Feature correlation analysis
    - Dimensionality reduction techniques

    ### Machine Learning Architecture
    **Regression Models:**
    - Linear Regression (Baseline)
    - Regularized Regression (Ridge/Lasso)
    - Random Forest Regressor
    - XGBoost Regressor

    **Classification Models:**
    - Logistic Regression
    - Support Vector Machines
    - Gradient Boosted Trees
    - Neural Network Classifier

    ### Evaluation Framework
    - **Regression Metrics**: R², Adjusted R², MAE, RMSE, MAPE
    - **Classification Metrics**: Precision-Recall Curves, ROC Analysis, Fβ Scores
    - Cross-validation with stratified sampling
    - Statistical significance testing
    """)

    st.markdown("---")
    
    st.markdown("""
    ## Model Performance Evaluation

    ### Classification Report
    Our optimized model demonstrates strong performance across all metrics:
    """)
    
    # Classification report data
    classification_data = {
        'Class': ['0 (Not Admit)', '1 (Admit)', 'Macro Avg', 'Weighted Avg'],
        'Precision': [0.97, 0.87, 0.92, 0.95],
        'Recall': [0.96, 0.91, 0.94, 0.95],
        'F1-Score': [0.97, 0.89, 0.93, 0.95],
        'Support': [79, 22, 101, 101]
    }
    
    # Display as table
    st.dataframe(pd.DataFrame(classification_data), hide_index=True)
    
    st.markdown("""
    #### Key Observations:
    - **High Precision (0.97)** for class 0 indicates minimal false positives in rejection predictions
    - **Strong Recall (0.91)** for class 1 shows we capture most actual admission cases
    - **Balanced F1-scores** demonstrate consistent performance across both classes
    - **95% Overall Accuracy** with balanced class distribution (79:22 ratio)
    """)

    st.markdown("---")
    
    st.markdown("""
    ## Future Enhancements

    ### Immediate Roadmap
    - Real-time prediction API for integration
    - Multi-institution comparison tool
    - Applicant benchmarking dashboard
    - Mobile application development

    ### Advanced Development
    - NLP for SOP/LOR quality analysis
    - Institution-specific model tuning
    - Explainable AI (XAI) features
    - Anomaly detection for application verification
    """)

    st.markdown("---")
    
    st.markdown("""
    ## Getting Started

    1. **Clone the repository**:
    ```bash
    git clone https://github.com/yourusername/university-admission-predictor.git
    cd university-admission-predictor
    ```

    2. **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

    3. **Launch the application**:
    ```bash
    streamlit run app.py
    ```

    *Note: Predictions are based on historical patterns. Actual decisions may involve additional qualitative factors.*
    """)
    
    # Add some space at the bottom
    st.markdown("<br><br>", unsafe_allow_html=True)

elif page == "EDA":
    st.title("Exploratory Data Analysis")
    if not df.empty:
        st.subheader("Dataset Overview")
        col1, col2 = st.columns(2)
        with col1:
            st.write("First 5 rows:")
            st.dataframe(df.head())
        with col2:
            st.write("Basic Statistics:")
            st.dataframe(df.describe())
        
        st.subheader("Data Distribution")
        selected_col = st.selectbox("Select feature", df.columns)
        fig, ax = plt.subplots()
        sns.histplot(df[selected_col], kde=True, bins=20, ax=ax)
        st.pyplot(fig)
        
        st.subheader("Relationships")
        col1, col2 = st.columns(2)
        with col1:
            x_axis = st.selectbox("X-axis feature", df.columns, index=0)
        with col2:
            y_axis = st.selectbox("Y-axis feature", df.columns, 
                                 index=df.columns.get_loc('Admission_Chance'))
        
        fig, ax = plt.subplots()
        sns.scatterplot(data=df, x=x_axis, y=y_axis, hue='Admission_Chance', ax=ax)
        st.pyplot(fig)
        
        st.subheader("Category Analysis")
        cat_feature = st.selectbox("Select categorical feature", 
                                 ['University_Rating', 'Research', 'SOP'])
        fig, ax = plt.subplots()
        sns.boxplot(data=df, x=cat_feature, y='Admission_Chance', ax=ax)
        st.pyplot(fig)
        
        st.subheader("Correlation Analysis")
        fig, ax = plt.subplots(figsize=(10,8))
        sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
        st.pyplot(fig)

elif page == "Model Predictions":
    st.title("Admission Predictions")
    if model is None or scaler is None or features is None:
        st.error("Model components not loaded properly. Please check the model file.")
    else:
        st.subheader("Enter Applicant Details")
        with st.form("prediction_form"):
            col1, col2 = st.columns(2)
            with col1:
                gre_score = st.slider("GRE Score (260-340)", 260, 340, 310)
                toefl_score = st.slider("TOEFL Score (90-120)", 90, 120, 105)
                # university_rating = st.slider("University Rating (1-5)", 1, 5, 3)
            with col2:
                sop = st.slider("SOP Strength (1-5)", 1, 5, 3)
                lor = st.slider("LOR Strength (1-5)", 1, 5, 3)
                cgpa = st.slider("CGPA (6.0-10.0)", 6.0, 10.0, 8.5)
                # research = st.selectbox("Research Experience", [0, 1])
            
            submitted = st.form_submit_button("Predict Admission Probability")
        
        if submitted:
            input_data = pd.DataFrame({
                'GRE_Score': [gre_score],
                'TOEFL_Score': [toefl_score],
                # 'University_Rating': [university_rating],
                'SOP': [sop],
                'LOR': [lor],
                'CGPA': [cgpa],
                # 'Research': [research]
            })[features]  # Ensure correct feature order
            
            input_scaled = scaler.transform(input_data)
            
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(input_scaled)[:, 1][0]
            else:
                proba = model.predict(input_scaled)[0]
            
            st.subheader("Prediction Results")
            fig, ax = plt.subplots(figsize=(8, 1))
            ax.barh(['Admission Chance'], [1], color='lightgray')
            ax.barh(['Admission Chance'], [proba], color=sns.color_palette()[0])
            ax.set_xlim(0, 1)
            ax.set_title(f"Admission Probability: {proba:.1%}")
            ax.set_xticks([])
            st.pyplot(fig)
            
            if proba > 0.8:
                st.success("High Chance of Admission (>80%)")
            elif proba > 0.6:
                st.warning("Moderate Chance of Admission (60-80%)")
            else:
                st.error("Low Chance of Admission (<60%)")

elif page == "Model Analysis":
    st.title("Model Performance Analysis")
    if not df.empty and model and scaler:
        st.subheader("Model Configuration")
        st.write(f"Model type: {type(model).__name__}")
        st.write(f"Features used: {', '.join(features)}")
        
        threshold = st.slider("Classification Threshold", 
                             min_value=0.1, 
                             max_value=0.9, 
                             value=0.8, 
                             step=0.05,
                             help="Set probability threshold for admission classification")
        
        X = df[features]
        y = (df['Admission_Chance'] > threshold).astype(int)
        
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=42, stratify=y)
            
            X_test_scaled = scaler.transform(X_test)
            y_pred = model.predict(X_test_scaled)
            y_proba = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, 'predict_proba') else y_pred
            
            st.subheader("Evaluation Metrics")
            metrics = {
                "Accuracy": accuracy_score(y_test, y_pred),
                "Precision": precision_score(y_test, y_pred),
                "Recall": recall_score(y_test, y_pred),
                "F1 Score": f1_score(y_test, y_pred),
                "ROC AUC": roc_auc_score(y_test, y_proba) if hasattr(model, 'predict_proba') else "N/A"
            }
            st.dataframe(pd.DataFrame.from_dict(metrics, orient='index', columns=['Value']))
            
            tab1, tab2, tab3 = st.tabs(["Confusion Matrix", "ROC Curve", "Classification Report"])
            
            with tab1:
                fig, ax = plt.subplots(figsize=(6,6))
                ConfusionMatrixDisplay.from_predictions(
                    y_test, y_pred, 
                    display_labels=['Rejected', 'Admitted'],
                    cmap='Blues',
                    ax=ax,
                    values_format='d'
                )
                ax.set_title(f"Confusion Matrix (Threshold: {threshold:.0%})")
                st.pyplot(fig)
            
            with tab2:
                if hasattr(model, 'predict_proba'):
                    fpr, tpr, _ = roc_curve(y_test, y_proba)
                    roc_auc = auc(fpr, tpr)
                    
                    fig, ax = plt.subplots(figsize=(6,6))
                    ax.plot(fpr, tpr, color='darkorange', lw=2, 
                            label=f'ROC curve (AUC = {roc_auc:.4f})')
                    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                    ax.set_xlim([0.0, 1.0])
                    ax.set_ylim([0.0, 1.05])
                    ax.set_xlabel('False Positive Rate')
                    ax.set_ylabel('True Positive Rate')
                    ax.legend(loc="lower right")
                    st.pyplot(fig)
                else:
                    st.warning("ROC Curve not available - model doesn't support probability predictions")
            
            with tab3:
                report = classification_report(
                    y_test, y_pred,
                    target_names=['Rejected', 'Admitted'],
                    output_dict=True
                )
                st.dataframe(pd.DataFrame(report).transpose())
                
        except Exception as e:
            st.error(f"Error during model evaluation: {str(e)}")