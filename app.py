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

# Set consistent style
plt.style.use('seaborn')
sns.set_palette("Blues_d")
sns.set_style("whitegrid")

# Set page configuration
st.set_page_config(page_title="University Admission Analysis", layout="wide")

# Define paths
BASE_DIR = r"C:\Users\HP\Desktop\Project _UAP\university_admission_predictor"
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
    st.image("https://images.unsplash.com/photo-1523050854058-8df90110c9f1?ixlib=rb-1.2.1&auto=format&fit=crop&w=1350&q=80", 
             use_container_width=True)
    st.markdown("""
    ## Welcome to the University Admission Prediction System
    
    This application predicts admission chances based on:
    - GRE Scores (260-340)
    - TOEFL Scores (90-120)
    - University Rating (1-5)
    - SOP Strength (1-5)
    - LOR Strength (1-5)
    - CGPA (6.0-10.0)
    - Research Experience (0 or 1)
    """)
    
    st.markdown("""
    **This smart application uses Machine Learning to analyze your academic records** and estimate your chances of getting admitted into universities.

    ---
    ###  How It Works:
    - Enter your **GRE**, **TOEFL**, and other academic info.
    - See the predicted **admission chance** instantly.
    - Explore **data insights**, **charts**, and **model evaluations**.

    ---
    ### Get Started:
    Select a page from the sidebar and begin your admission journey!

    ---
    """,
    unsafe_allow_html=True
        )

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
        with st.expander("Feature Distributions"):
            selected_col = st.selectbox("Select feature", df.columns)
            fig, ax = plt.subplots()
            sns.histplot(df[selected_col], kde=True, bins=20, ax=ax)
            st.pyplot(fig)
        
        st.subheader("Relationships")
        with st.expander("Scatter Plots"):
            col1, col2 = st.columns(2)
            with col1:
                x_axis = st.selectbox("X-axis feature", df.columns, index=0)
            with col2:
                y_axis = st.selectbox("Y-axis feature", df.columns, 
                                     index=df.columns.get_loc('Admission_Chance'))
            
            fig, ax = plt.subplots()
            sns.scatterplot(data=df, x=x_axis, y=y_axis, hue='Admission_Chance', ax=ax)
            st.pyplot(fig)
        
        with st.expander("Category Analysis"):
            cat_feature = st.selectbox("Select categorical feature", 
                                     ['University_Rating', 'Research', 'SOP'])
            fig, ax = plt.subplots()
            sns.boxplot(data=df, x=cat_feature, y='Admission_Chance', ax=ax)
            st.pyplot(fig)
        
        st.subheader("Correlation Analysis")
        with st.expander("Correlation Heatmap"):
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
                university_rating = st.slider("University Rating (1-5)", 1, 5, 3)
            with col2:
                sop = st.slider("SOP Strength (1-5)", 1, 5, 3)
                lor = st.slider("LOR Strength (1-5)", 1, 5, 3)
                cgpa = st.slider("CGPA (6.0-10.0)", 6.0, 10.0, 8.5)
                research = st.selectbox("Research Experience", [0, 1])
            
            submitted = st.form_submit_button("Predict Admission Probability")
        
        if submitted:
            input_data = pd.DataFrame({
                'GRE_Score': [gre_score],
                'TOEFL_Score': [toefl_score],
                'University_Rating': [university_rating],
                'SOP': [sop],
                'LOR': [lor],
                'CGPA': [cgpa],
                'Research': [research]
            })[features]  # Ensure correct feature order
            
            input_scaled = scaler.transform(input_data)
            
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(input_scaled)[:, 1][0]
            else:
                proba = model.predict(input_scaled)[0]
            
            st.subheader("Prediction Results")
            fig, ax = plt.subplots(figsize=(8, 1))
            ax.barh(['Admission Chance'], [1], color='lightgray')
            ax.barh(['Admission Chance'], [proba], color='#1f77b4')
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