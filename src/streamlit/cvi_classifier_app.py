import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Load necessary data - Still needs to be figured out
#df = pd.read_csv("HSOS_Market_merged_data.csv")  # Update with actual path
#models = {
    #"Random Forest": joblib.load("random_forest_optimized.pkl"),
    #"SVM": joblib.load("svm_model_optimized.pkl"),
    #"XGBoost": joblib.load("xgboost_model_optimized.joblib"),
#}

# Streamlit App Title
st.set_page_config(page_title="CVI - Classifier", layout="wide")

# Sidebar Navigation
st.sidebar.title("Navigation")
pages = ["Home", "EDA", "Modeling", "Prediction", "Perspectives", "About"]
selection = st.sidebar.radio("Go to", pages)

# 1. HOME PAGE
if selection == "Home":
    st.title("CVI - Classifier")
    st.subheader("Context and Objectives")
    st.write("""
    - **Problem Need**: Addressing humanitarian vulnerability in Syria.
    - **Objectives**: Build a classifier to assist organizations in decision-making.
    - **Framework**: Machine Learning applied to the HSOS dataset.
    """)

# 2. EDA PAGE
elif selection == "EDA":
    st.title("Exploratory Data Analysis")
    st.subheader("Key Visualizations")
    
    # Example visualizations (replace with relevant ones)
    fig, ax = plt.subplots()
    sns.histplot(df["feature_x"], bins=30, kde=True, ax=ax)  # Replace with relevant feature
    st.pyplot(fig)

# 3. MODELING PAGE
elif selection == "Modeling":
    st.title("Model Performance")
    st.subheader("Selected Models and Metrics")
    
    # Placeholder for model results (update with actual metrics)
    st.write("""
    - **Random Forest**: Accuracy - 85%
    - **SVM**: Accuracy - 83%
    - **XGBoost**: Accuracy - 87%
    """)

# 4. PREDICTION PAGE
elif selection == "Prediction":
    st.title("Make a Prediction")
    
    governorate = st.selectbox("Select Governorate", df["governorate"].unique())
    location = st.selectbox("Select Location", df[df["governorate"] == governorate]["location"].unique())
    
    if st.button("Predict"):
        subset = df[(df["governorate"] == governorate) & (df["location"] == location)]
        probabilities = {model: models[model].predict_proba(subset)[:, 1].mean() for model in models}
        st.write("Predicted Vulnerability Scores:")
        st.json(probabilities)

# 5. PERSPECTIVES PAGE
elif selection == "Perspectives":
    st.title("Perspectives and Insights")
    st.write("Discussion on model outcomes and recommendations.")

# 6. ABOUT PAGE
elif selection == "About":
    st.title("About the Project")
    st.write("""
    - **Team Members**: [Names]
    - **Data Source**: IMPACT
    """)


