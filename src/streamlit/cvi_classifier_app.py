import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

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
    ## Context
             
    In humanitarian **crises**, international non-governmental **organizations** (NGOs) face significant **challenges** in identifying needs and targeting **communities** due to the complexity and volume of **data**. 
    Despite the availability of large datasets, information is often underutilized, failing to transform into actionable insights accessible to NGOs.

    Syria, experiencing one of the largest humanitarian crises in recent decades, presents a complex ecosystem of NGOs supporting communities. The availability of humanitarian data in Syria is substantial compared to other conflict zones, making it a suitable case for this project.

    st.image("src/streamlit/images/1_country-syria.png", caption="Humanitarian Data in Syria")

    NGOs such as [UNHCR](https://www.unhcr.org) and [IMPACT REACH](https://www.impact-initiatives.org/where-we-work/syria/) Initiatives have built a robust **database** over the past six to seven years, updated monthly, describing thousands of **communities** with internally displaced people (IDPs).
    """)

    st.write("""
    ## Problematic
    Despite the availability of extensive **humanitarian data**, NGOs struggle to extract **actionable insights** due to the **complexity** of datasets and lack of systematic **aid distribution**. Large-scale community **displacements** require rapid response mechanisms, yet existing data processing methods are inadequate in prioritizing and allocating **resources** effectively.
    """)

    st.write("""
    ## Objectives
    - **Develop a classifier** that predicts **vulnerability trends**
    - **Assist NGOs** in prioritizing **aid distribution** effectively
    - **Utilize multi-label classification** to analyze different forms of **assistance**

    The primary goal is to transform humanitarian **data** (monthly time series spanning five years) into a **predictive** and **classifier model**. By learning from historical trends, the **model** will forecast future **vulnerability**, aiding decision-makers in prioritizing aid and planning interventions.
    """)

    st.write("""
    ## Data
    The project primarily relies on the **Humanitarian Situation Overview in Syria (HSOS)** dataset, which provides monthly data from **2019 to 2024** for **Northeast Syria (NES)** and **Northwest Syria (NWS)**.
    
    Additionally, data from the **Joint Market Monitoring Initiative (JMMI)** is integrated, focusing on the **Survival Minimum Expenditure Basket (SMEB)**, which captures price trends, regional variations, and relationships between essential expenditure categories. Initiated by the **Cash Working Group**, the JMMI dataset aids in understanding market dynamics and challenges across Syria.
    
    For consistency, the analysis is restricted to data from **2021 to 2023**, consolidating all regional time points into a single dataset. The final dataset includes **21 indicators** and over **4,700 observations**.
    """)

    st.write("""
    ## Framework
    The **HSOS dataset** contains monthly reports from **2019 to 2024**, detailing information across **11 indicator groups**:
    1. **Demographics**
    2. **Shelter**
    3. **Electricity & Non-Food Items (NFIs)**
    4. **Food Security**
    5. **Livelihoods**
    6. **Water, Sanitation, and Hygiene (WASH)**
    7. **Health**
    8. **Education**
    9. **Protection**
    10. **Accountability & Humanitarian Assistance**
    11. **Priority Needs**
    
    A total of **110 HSOS files** were initially analyzed. To maintain **data consistency**, the dataset was limited to reports from **2021 to 2023**, reducing the selection to **61 files**. Since no standardized catalog of questions exists across files, an **indicator selection process** was applied, retaining only those present in at least **57 of the 61 files**. The final dataset consists of **1,668 indicators** and **52,095 observations**.
    
    The **JMMI dataset**, focusing on the **Survival Minimum Expenditure Basket (SMEB)**, complements the HSOS data by analyzing economic conditions and market trends. The dataset originally contained **123 columns**, detailing item-specific prices. To streamline the analysis, individual item prices were removed in favor of **aggregated** subtotals, totals, and categorical variables.
    
    After restricting data to the years **2021 to 2023**, the final **JMMI dataset** includes **21 indicators** and over **4,700 observations**, ensuring alignment with the HSOS dataset.
    """)

# 2. EDA PAGE
elif selection == "EDA":
    st.title("Exploratory Data Analysis")
    st.subheader("Key Visualizations from Report 1")

    st.write("""
    This section presents exploratory data analysis using the HSOS dataset. 
    Key highlights:
    - **Distribution of humanitarian assistance types**
    - **Demographic characteristics of communities**
    - **Market trends impacting aid delivery**
    """)

    # Placeholder for visuals - Replace with actual visualizations
    st.write("🚧 Visualization loading setup in progress...")

# 3. MODELING PAGE
elif selection == "Modeling":
    st.title("Model Performance")
    st.subheader("Selected Models and Metrics")

    st.write("""
    The following machine learning models were tested:
    - **Random Forest**: Accuracy - 95%
    - **SVM**: Accuracy - 94%
    - **XGBoost**: Accuracy - 96%
    
    Further optimizations included:
    - **Hyperparameter tuning** (GridSearch & BayesSearch).
    - **Feature importance analysis** using SHAP.
    - **Per-class threshold tuning** to improve recall for underrepresented labels.
    """)

# 4. PREDICTION PAGE
elif selection == "Prediction":
    st.title("Make a Prediction")
    st.write("🚧 Prediction functionality will be implemented soon.")

# 5. PERSPECTIVES PAGE
elif selection == "Perspectives":
    st.title("Perspectives and Insights")
    st.write("""
    ## Findings from Report 3
    - **XGBoost + Per-Class Thresholding** provided the best predictive performance.
    - **SMOTE** slightly improved minority class recall but increased false positives.
    - **SHAP analysis** highlighted key drivers of humanitarian assistance needs.

    ## Recommendations:
    - Focus on **feature selection** to improve generalization.
    - Explore **threshold optimization** to balance recall vs. precision.
    - Consider **deep learning architectures** for further improvement.
    """)

# 6. ABOUT PAGE
elif selection == "About":
    st.title("About the Project")
    st.write("""
    ## Project Contributors
    - **Ghiath Al Jebawi** - Data Scientist (Humanitarian focus)
    - **Bercin Ersoz** - Statistician (Data Management)
    - **Allaeldene Ilou** - Data Engineer / ML Ops
    - **Caspar Stordeur** - Data Transformation & Analysis

    ## Data Source:
    - **IMPACT REACH Initiative**
    - **HSOS Dataset (Humanitarian Situation Overview in Syria)**
    - **JMMI Dataset (Joint Market Monitoring Initiative)**
    """)
