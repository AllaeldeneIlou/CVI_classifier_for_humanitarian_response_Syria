import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split
import numpy as np

# Define the dynamic path for images relative to the script's working directory
image_dir = Path(__file__).parent / "images"

# Streamlit App Title
st.set_page_config(page_title="CVI - Classifier", layout="wide")

# Sidebar Navigation
st.sidebar.title("Navigation")
pages = ["Home", "EDA", "Modeling", "Prediction", "Perspectives", "About"]
selection = st.sidebar.radio("Go to", pages)

# 1. HOME PAGE
if selection == "Home":
    # Create two columns: text on the left (70%) and image on the right (30%)
    col1, col2 = st.columns([2, 1])
    with col1:
        st.title("CVI - Classifier")
        st.markdown("<u>Multi-Label Classification for Humanitarian Assistance Analysis</u>", unsafe_allow_html=True)

    st.write("""
    ## Context
             
    In humanitarian **crises**, international non-governmental organizations **(NGOs)** face significant **challenges** in identifying needs and **targeting** communities due to the complexity and volume of **data**. 
    Despite the availability of large datasets, information is often underutilized, failing to transform into actionable insights accessible to NGOs.

    Syria, experiencing one of the largest humanitarian crises in recent decades, presents a complex ecosystem of NGOs supporting communities. The availability of humanitarian data in Syria is substantial compared to other conflict zones, making it a suitable case for this project.

    NGOs such as [UNHCR](https://www.unhcr.org) and [IMPACT REACH](https://www.impact-initiatives.org/where-we-work/syria/) Initiatives have built a robust **database** over the past six to seven years, updated monthly, describing thousands of **communities** with internally displaced people **(IDPs)**.
    """)

    with col2:
        # Display the image on the right
        image_path = Path(__file__).parent / "images/1_country-syria.png"
        st.image(str(image_path), use_container_width=True)

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
    """)

    # Markdown bullet points
    st.markdown("""
        - **Demographics**
        - **Shelter**
        - **Electricity & Non-Food Items (NFIs)**
        - **Food Security**
        - **Livelihoods**
        - **Water, Sanitation, and Hygiene (WASH)**
        - **Health**
        - **Education**
        - **Protection**
        - **Accountability & Humanitarian Assistance**
        - **Priority Needs**
    """)


    # Additional Description Below
    st.write("""    
    A total of **110 HSOS files** were initially analyzed. To maintain **data consistency**, the dataset was limited to reports from **2021 to 2023**, reducing the selection to **61 files**. Since no standardized catalog of questions exists across files, an **indicator selection process** was applied, retaining only those present in at least **57 of the 61 files**. The final dataset consists of **1,668 indicators** and **52,095 observations**.
    
    The **JMMI dataset**, focusing on the **Survival Minimum Expenditure Basket (SMEB)**, complements the HSOS data by analyzing economic conditions and market trends. The dataset originally contained **123 columns**, detailing item-specific prices. To streamline the analysis, individual item prices were removed in favor of **aggregated** subtotals, totals, and categorical variables.
    
    After restricting data to the years **2021 to 2023**, the final **JMMI dataset** includes **21 indicators** and over **4,700 observations**, ensuring alignment with the HSOS dataset.
    """)

# 2. EDA PAGE
elif selection == "EDA":
    st.title("EDA")

    # Define a function for displaying images with captions, standard size, and centering
    def display_image_with_caption(image_path, caption):
        st.image(str(image_path), use_container_width=True)  # Use the recommended parameter

        # Centering the caption using Markdown & HTML
        st.markdown(
            f'<p style="text-align: center; font-size: 14px; color: gray;">{caption}</p>',
            unsafe_allow_html=True
            )
    
    # Create tabs for different categories of analysis
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Humanitarian Assistance", "Demographics", "Data Quality Assessment", 
        "Feature Analysis", "Market Trends"
    ])
    
    # Tab 1: Humanitarian Assistance
    with tab1:
        st.subheader("Humanitarian Assistance Provided")
        st.write("""
        This section explores the distribution and trends of humanitarian aid types 
        provided to internally displaced persons (IDPs). The following visualizations 
        illustrate the **types of assistance**, their **distribution**, and **correlations** 
        with other variables.
        """)

        st.write("""
        This **pie chart** illustrates the percentage distribution of reported humanitarian assistance provided to IDP households in the last 30 days. 
        **Food & Nutrition (28.4%)** is the most common type of aid, while **NA - No humanitarian assistance reported (26.9%)** highlights gaps in support. 
        Other categories include **Health (15.8%)**, **WASH (10%)**, and Cash **assistance (8.4%)**, reflecting diverse aid priorities. The **Other category (10.4%)** aggregates responses below 5%.
        """)
        display_image_with_caption(image_dir / "1_Percentage_distribution_Humanitarian_assistance_to_IDP_households_community_last_30_days.png", 
                                   "Distribution of different types of humanitarian assistance received by IDP households.")
        
        st.write("""
        Cramérs V **Correlation Matrix** - Binary targets (Humanitarian assistance categories) showed no **multicollinearity**, confirming their independence
        """)
        display_image_with_caption(image_dir / "2_Correlation_Matrix_Target_Variable.png", 
                                   "Cramér's V Correlation Matrix.")
        
        st.write("""
        This **stacked bar chart** presents the monthly percentage distribution of humanitarian assistance provided to IDP households over the past 30 days.
        **Food & Nutrition** remains the dominant category across most months, while other assistance types, such as **Health** and **WASH**, show noticeable fluctuations.
        Peaks in **WASH** support may reflect seasonal variations or infrastructure challenges.
        The visualization highlights **dynamic shifts** in aid distribution and the influence of external factors on humanitarian priorities.
        """)
        col1, col2, col3 = st.columns([1, 2, 1])  # Adjust the ratio to control centering
        with col2:
            st.image(image_dir / "3_Monthly_percentage_distribution_humanitarian_assistance_provided_to_IDP.png", caption= "Monthly trend of humanitarian assistance received by IDPs from 2021 to 2023.", width=int(0.7 * 1000))
    
    # Tab 2: Demographics
    with tab2:
        st.subheader("Demographic Analysis")
        st.write("""
        The following charts highlight the geographic distribution of assessed populations 
        across governorates and districts.
        """)
        
        st.markdown(
            "This **first pie chart illustrates the percentage distribution of the assessed population across governorates.** "
            "Idleb and Aleppo dominate, with similar proportions (**31.3%** and **30.8%**, respectively), while **Deir-ez-Zor** and other smaller regions contribute minimally to the dataset."
        )

        st.image(image_dir / "4_Demographic_distribution_across_governorate.png", caption="Demographic distribution across governorates", width=int(0.5 * 800))
      
        st.markdown(
            "This **second pie chart shows the distribution across districts.** "
            "A significant proportion (**36.1%**) is categorized as **'Other'** (sum of categories below 5%), followed by **Ar-Raqqa (10.9%)** and **Harim (9.7%)**. "
            "This highlights the geographic spread and variability in representation at the district level, which includes **23 unique values**."
        )
        
        st.image(image_dir / "5_Percentage_distribution_districts.png", caption="Percentage distribution across districts", width=int(0.8 * 800))

    # Tab 3: Data Quality Assessment
    # Missing Data
    with tab3:
        st.subheader("Missing Value Analysis")
        st.write("""
        A **high proportion of missing data** necessitates careful **feature selection**. A **20% missing value threshold** balances **data retention** and **noise reduction**, 
        but it can be adjusted based on model needs. **Feature selection** should be applied **after train-test splitting** to prevent **data leakage** and improve model robustness.
        """)
        col1, col2, col3 = st.columns([1, 2, 1])  # Adjust the ratio to control centering
        with col2:
            st.image(image_dir / "Missing_value_analysis.png", caption=None, width=int(0.8 * 800))
       
        st.subheader("Time-Series Analysis Insights")

        st.markdown(
            "**Time-series analysis** revealed **seasonal trends** in support distribution. "
            "**Months were One-Hot Encoded** as features to capture **temporal patterns**."
            )

        st.markdown(
            "The **Augmented Dickey-Fuller test** showed **mixed stationarity**:"
            "\n- **Health & Cash Assistance**: **Stationary** (p-values below significance level)."
            "\n- **Shelter & Explosive Hazard Awareness**: **Non-stationary** (failed to reject null hypothesis)."
            )

        st.markdown(
            "Given the **short timeframe**, **time-series modeling** was not pursued. "
            "Since the focus is on **classification**, including **months as features** is sufficient."
            )
        
        st.subheader("Class Imbalance in Assistance Data")

        st.markdown(
            "The **dataset is unbalanced**, with most assistance categories dominated by **'0s'** (no assistance). "
            "This **imbalance risks biased learning**, where models **favor the majority class** and fail to detect **critical minority cases ('1s')**."
            )

        st.markdown(
            "To **handle class imbalance**, consider:"
            "\n- **Class-weighted algorithms** (e.g., **Gradient Boosting**, **Random Forests**)."
            "\n- **Oversampling techniques** (e.g., **SMOTE**)."
            "\n- **Adjusting classification thresholds** to **prioritize recall** for minority cases."
            )
        col1, col2, col3 = st.columns([1, 2, 1])  # Adjust the ratio to control centering
        with col2:
            st.image(str(image_dir / "value_counts_table.png"), caption=None, width=int(1 * 1000))

    # Tab 4: Feature Analysis
    with tab4:
        st.subheader("Feature Importance and Relationships")
        st.markdown(
        "To analyze **priority needs**, three key questions (**Top 1, Top 2, Top 3**) were **aggregated** using weighted values:"
        "\n- **Top 1** → **Weight: 3**"
        "\n- **Top 2** → **Weight: 2**"
        "\n- **Top 3** → **Weight: 1**"
        )
    
        st.markdown(
        "The data was **reshaped** (melted) and **aggregated monthly** to track **trends over time**. "
        "The **top five priority needs**—**Food, Health, Livelihoods, Shelter, and WASH**—were identified and plotted as a time series."
        )

        st.markdown(
        "The graph shows the **weighted monthly distribution** of the **top five needs**:"
        "\n- **Food** remains the dominant priority."
        "\n- **Livelihoods & Shelter** follow as critical needs."
        "\n- **Health & WASH** show smaller but variable shares."
        )

        st.markdown(
        "These **priority needs** likely have a **strong relationship with the target variable**, "
        "as addressing them **directly impacts humanitarian outcomes** for **IDP populations**."
        )
        col1, col2, col3 = st.columns([1, 2, 1])  # Adjust the ratio to control centering
        with col2:
            st.image(str(image_dir / "6_Weighted_monthly_percentage_distribution_top_5_Priority.png"), caption=None, width=int(0.5 * 1000))

        st.subheader("Challenges in Accessing Humanitarian Assistance")
    
        st.markdown(
        "The **pie chart** highlights **challenges in accessing humanitarian assistance**:"
        "\n- **No assistance reported (22.6%)** → No challenges recorded."
        "\n- **Insufficient quantity (20.6%)** → Aid does not meet demand."
        "\n- **Irrelevant assistance (15.8%)** → Provided aid does not match needs."
        "\n- **Not meeting eligibility (8.1%)** → Access restrictions."
        )

        st.markdown(
        "These factors **affect aid distribution** and **highlight gaps in tailored assistance** for **IDP communities**."
        )
        col1, col2, col3 = st.columns([1, 2, 1])  # Adjust the ratio to control centering
        with col2:
            st.image(str(image_dir / "7_Percentage_Distribution_missing_information.png"))
    
        st.subheader("Preferred Methods of Information Delivery")
    
        st.markdown(
        "The **pie chart** illustrates the **preferred methods** of providing information to households. "
        "**Personal face-to-face** is the most preferred method (**32.4%**), followed by **WhatsApp or other mobile platforms** (**28.5%**). "
        "The **effectiveness of these methods** may influence the **target variable**, impacting **access to aid**."
        )
        col1, col2, col3 = st.columns([1, 2, 1])  # Adjust the ratio to control centering
        with col2:
            st.image(str(image_dir / "8_Preferred_Information_Methods.png"))
    
        # Relationship between target variable and features
        st.subheader("Relationship Between Target Variable and Features")
    
        st.markdown(
            "**Cramer's chi-square test** was performed to capture **non-linear relationships**. "
            "The table below shows the **number of significant (p-value < 0.05) and insignificant features** based on the test statistics."
        )
    
        col1, col2, col3 = st.columns([1, 2, 1])  # Adjust the ratio to control centering
        with col2:
            st.image(str(image_dir / "Relationshi_target_features_Cramer.png"))
    
        st.markdown(
            "Analysis of **humanitarian assistance categories** reveals varying **predictive power**. "
            "**Shelter** and **Food/Nutrition** show a **higher number of important feature relationships**, indicating **strong associations** with other variables. "
            "In contrast, **Agricultural Supplies** and **Explosive Hazard Risk Awareness** have a **higher proportion of unimportant relationships**, suggesting **weaker connections**."
        )
    
        st.markdown(
            "These patterns highlight the **complexity of humanitarian data**. Some categories are **strongly tied to contextual factors** (e.g., economic conditions, shelter dynamics), while others are more **situational or region-specific**. "
            "High chi-square values confirm **statistically significant relationships** for some categories, while others have **negligible influence**."
        )
    
        st.markdown(
            "This underscores the need for **targeted modeling approaches**. "
            "Filtering **irrelevant features** for weaker categories and **prioritizing key variables** for strongly associated categories can **enhance model performance**."
        )

    # Tab 5: Market Trends
    with tab5:
        st.subheader("Joint Market Monitoring Initiative (JMMI) Analysis")
    
        st.markdown(
        "The Joint Market Monitoring Initiative **(JMMI)** tracks the Survival Minimum Expenditure Basket **(SMEB)** across **Syrian regions**, "
        "analyzing **price trends, regional variations, and SMEB component relationships**."
        )
    
        st.markdown(
        "Developed by the Cash Working Group **(CWG)**, JMMI assesses **market functionality** by **monitoring monthly prices and stock levels** "
        "of essential commodities to support **cash-based responses** and **food assistance programs**. Data is collected across multiple **governorates** through **retailer monitoring** in key markets. SMEB components, represent the **minimum culturally-adjusted needs** for a **six-person household** in Syria per month. "
        )
    
        st.markdown(
        "This section analyzes **SMEB total costs** and its **components**—**food, non-food items, cooking fuels, water, and mobile data**. "
        "To **prevent multicollinearity**, only **component subtotals** are included in the model."
        )

        st.markdown(
            "The **pie chart** shows that food dominates the SMEB calculations. Over **80%** of the component analysis is **food**, followed by **non-food items**, **cooking fuels**, and **water**."
        )

        col1, col2, col3 = st.columns([1, 2, 1])  # Adjust the ratio to control centering
        with col2:
            st.image(str(image_dir / "9_Market_Overall_Distribution_SMEB_Components.png"), caption=None, width=int(0.4 * 800))

        st.subheader("SMEB Cost and Component Trends")
        st.markdown(
        "The **graph** shows SMEB cost and component trends over time in **Northeast and Northwest Syria**."
        )
        
        st.markdown(
        "The **time series analysis** reveals a **similar pattern** in both regions:"
        "\n- **Food** has the **largest share** and follows an **upward trend until mid-2023**, when **prices drop sharply**."
        "\n- Other components show a **smoother, less fluctuating trend**."
        "\n- The **Syrian Pound/USD exchange rate** used is **fixed**, not floating."
        )

        st.markdown(
        "The **sharp drop in July 2023** (due to an **80% devaluation**) is the **primary cause** of this price shift."
        "\n- The **Northeast** experiences **significant volatility**, with a **sharp mid-2023 decline** driven by currency devaluation."
        "\n- The **Northwest** is **more stable**, with **SMEB costs remaining consistent** over time."
        )    
        st.image(str(image_dir / "10_timeseries_SMEB_components.png"))
        
        st.subheader("Correlation and Multicollinearity")

        st.markdown(
        "The **correlation matrix** reveals a **strong positive correlation** between **food and non-food items**, indicating **multicollinearity**. "
        "**Variance Inflation Factor (VIF) analysis** confirms this issue."
        )

        st.markdown(
        "**Multicollinearity** can lead to **overfitting** by introducing **redundant information** in **high-dimensional datasets**. "
        "A potential **solution** is to **exclude NFI values** during modeling to improve estimation accuracy."
        )
        col1, col2, col3 = st.columns([1, 2, 1])  # Adjust the ratio to control centering
        with col2:
            st.image(str(image_dir / "11_Matrix_SMEB_Components.png"), caption=None, width=int(0.6 * 800))


        st.markdown(
        "**High VIF values** for **price_smeb_food** and **price_smeb_nfi** indicate **significant multicollinearity**, "
        "suggesting **redundancy** in regression models. These variables have a **strong correlation (0.92)**."
        )

        st.markdown(
        "To **prevent overfitting**, **NFI prices** are **excluded from the dataset**."
        )


# 3. MODELING PAGE
elif selection == "Modeling":

    # Title 0 (Biggest)
    st.title("MODELING")

    # Title 1 
    st.header("Data Preprocessing & Feature Engineering")
    st.write("""
    - **Categorical Encoding:** One-Hot Encoding was applied to (Governorate, Month, etc.), while Frequency Encoding was used for District to reduce dimensionality.
    - **Feature Scaling:** StandardScaler was applied to price-related columns, and Min-Max normalization was used for numerical features exceeding 1 to ensure consistency across scales.
    - **Currency Standardization:** Prices from different regions were converted to USD.
    - **Feature Selection:** The dataset initially contained 975 features. After applying a 20% missing value threshold, it was reduced to 789. Further reduction to 497 features was achieved by removing low-variance features (threshold = 0.01).
    - **Handling Missing Values:** Missing values were imputed using KNN Imputation (k=5) to preserve the underlying data distribution.
    """)

    # Title 2 Sections
    st.header("Multi Output Classification")
    st.write(
        "The ML type is a MultiOutputClassifier to handle a multi-label classification problem, "
        "where each instance may belong to multiple categories simultaneously "
        "(different categories of humanitarian assistance needed)."
    )

    st.header("Performance Metrics")
    st.write(
        "- **Hamming Loss**: Directly measures the fraction of misclassified labels in multi-label classification.\n"
        "- **F1-Score (Macro & Micro)**: Evaluates balance between precision and recall across all labels."
    )

    st.header("Model Choice & Optimization")
    st.subheader("Initial Models")
    st.write(
        "We evaluated multiple models to establish baseline performance and identify the most effective approach: "
        "Logistic Regression, Random Forest, SVM, Naive Bayes, Decision Tree, KNN all wrapped with a MultiOutputClassifier."
    )

    # table 13
    data = [
        ["Logistic Regression", 0.0293, 0.5, 0.77, 0.6922],
        ["Random Forest", 0.0216, 0.44, 0.81, 0.7597],
        ["Naive Bayes", 0.1007, 0.31, 0.5, 0.384],
        ["SVM", 0.0304, 0.49, 0.77, 0.6859],
        ["KNN", 0.0261, 0.54, 0.8, 0.6913],
        ["Decision Tree", 0.031, 0.49, 0.77, 0.6868]
    ]
    columns = ["Model", "Hamming Loss", "F1-Score (Macro)", "F1-Score (Micro)", "Accuracy"]
    st.table(pd.DataFrame(data, columns=columns))
    st.markdown("<p style='text-align: center; font-style: italic;'>Table 13 <i>MultiTarget</i> - Base Models</p>", unsafe_allow_html=True)

    st.subheader("GridSearchCV Hyperparameter tuning results")
    st.write("To optimize model performance, GridSearchCV (3-fold cross-validation) was applied to Random Forest, SVM, and XGBoost.")

    # table 14
    data = [
        ["Random Forest", 0.021799, 0.456789, 0.825133, 0.758013],
        ["SVM", 0.020194, 0.592332, 0.846237, 0.771229],
        ["XGBoost", 0.019463, 0.599793, 0.850648, 0.783381]
    ]
    columns = ["Model", "Hamming Loss", "F1-Score (Macro)", "F1-Score (Micro)", "Accuracy"]
    st.table(pd.DataFrame(data, columns=columns))
    st.markdown("<p style='text-align: center; font-style: italic;'>Table 14 <i>MultiTarget</i> - Hyperparameter Tuning Models</p>", unsafe_allow_html=True)

    st.write("Based on the results, XGBoost and SVM demonstrated superior performance, leading to their selection for ensemble methods.")
    st.write("XGBoost consistently delivers superior results due to its ability to efficiently handle large feature spaces while maintaining computational efficiency. Its robust performance on imbalanced datasets makes it particularly well-suited for complex classification tasks.")
    st.write("Additionally, XGBoost leverages parallelization, significantly reducing training time compared to traditional boosting methods. Another key advantage is its built-in regularization mechanisms, which help prevent overfitting, ensuring better generalization on unseen data.")

    st.header("Ensemble Models (Stacking & Voting)")
    st.subheader("Stacking Classifier")
    st.write("Chosen Base Models: Best XGBoost and SVM models (trained separately for each label).")
    st.write("Meta Model: Logistic Regression (selected for its generalization capability).")
    st.write("Implementation: Separate stacking classifiers were trained for each label using StackingClassifier, ensuring each label had its own specialized ensemble.")
    st.subheader("Voting Classifier (Hard & Soft Voting)")
    st.write("Soft voting (best performance) didn't significantly outperform other models, meaning its contribution to overall predictive performance is marginal")

    # Table 15
    data = [
        ["Stacking", 0.019311, 0.587003, 0.852073, 0.78171],
        ["Hard Voting", 0.020052, 0.562929, 0.841465, 0.773204],
        ["Soft Voting", 0.01914, 0.596515, 0.85289, 0.783837]
    ]
    columns = ["Model", "Hamming Loss", "F1-Score (Macro)", "F1-Score (Micro)", "Accuracy"]
    st.table(pd.DataFrame(data, columns=columns))
    st.markdown("<p style='text-align: center; font-style: italic;'>Table 15 <i>MultiTarget</i> - Voting Classifier</p>", unsafe_allow_html=True)

    st.header("SMOTE (XGBoost, SVM, Random Forest)")
    st.write("Following the methodology from the thesis \"Handling Data Imbalance in Multi-label Classification\" by Sukhwani (2020), we know that traditional SMOTE is not directly compatible with multi-label problems, so MLSMOTE was used to oversample minority labels.")

    # Table 16
    data = [
        ["XGBoost", 0.02107, 0.59, 0.84, 0.7653],
        ["SVM", 0.02047, 0.59, 0.84, 0.7691],
        ["Random Forest", 0.02294, 0.44, 0.81, 0.7448]
    ]
    columns = ["Model", "Hamming Loss", "F1-Score (Macro)", "F1-Score (Micro)", "Accuracy"]
    st.table(pd.DataFrame(data, columns=columns))
    st.markdown("<p style='text-align: center; font-style: italic;'>Table 16 <i>MultiTarget</i> - MLSMOTE Results</p>", unsafe_allow_html=True)

    st.write("SMOTE slightly improved the micro F1-score, indicating better performance across all labels, especially for minority classes. However, the drop in macro F1-score suggests that predicting minority classes remains a challenge. Given that overall accuracy decreased slightly and the improvements in minority classes were minimal, it seems more practical to proceed with the original models combined with feature selection rather than relying on SMOTE.")

    st.header("Feature importance analysis using XGBoost")
    st.write("The experiment aimed at identifying redundant features that could be removed for better efficiency.")

    # Figure 09
    image_dir = Path(__file__).parent / "pics"
    st.image(image_dir / "Figure 9.png", caption="Figure 9 MultiTarget - Feature Importance")

    # Table 17
    data = [
        ["XGBoost (Thr: 0.001)", 340, 0.019615, 0.580884, 0.849219, 0.780799],
        ["XGBoost (Thr: 0.005)", 33, 0.035024, 0.419686, 0.725623, 0.674009],
        ["XGBoost (Thr: 0.01)", 6, 0.049484, 0.088401, 0.54055, 0.615069]
    ]
    columns = ["Model", "Number of Selected Features", "Hamming Loss", "F1-Score (Macro)", "F1-Score (Micro)", "Accuracy"]
    st.table(pd.DataFrame(data, columns=columns))
    st.markdown("<p style='text-align: center; font-style: italic;'>Table 17 <i>MultiTarget</i> - Feature Selection & Performance</p>", unsafe_allow_html=True)

    st.write("The table presents the impact of different feature importance thresholds on model performance. As the threshold increases, the number of selected features decreases significantly—from 340 at 0.001 to just 6 at 0.01.")
    st.write("Given this analysis, continuing with the 340 selected features is the optimal approach, as it retains strong predictive power while reducing dimensionality compared to using all features.")

    # Section: Deep Learning Models
    st.markdown("## **Deep Learning Models**")
    st.write("""
    We experimented with **MLP, TabNet, Autoencoder + MLP, and CNN** for multi-label classification.  
    """)

    st.markdown("""
    ### **Optimization & Challenges**
    - **Hyperparameter tuning** improved MLP performance.  
    - Deep learning models **struggled with interpretability and efficiency**.
    - Despite capturing **complex patterns**, they were **less effective** for tabular data due to:
    - **High computational cost**.
    - **Sensitivity to hyperparameters**.
    """)

    st.markdown("""
    ### **Final Decision**
    - **XGBoost** outperformed deep learning models.
    - Its **tree-based structure** handled tabular data more efficiently.
    - It provided **better interpretability** and **stronger performance**.
    """)

    # Deep Learning Model Results - Table 13
    data_dl = [
        ["MLP", 0.02508, 0.47954, 0.80345, 0.73583],
        ["MLP Tuned", 0.02397, 0.48027, 0.81587, 0.73963],
        ["TabNET", 0.02856, 0.32815, 0.77748, 0.71669],
        ["Autoencoder + MLP", 0.02759, 0.43861, 0.77921, 0.71669],
        ["CNN Initial", 0.09867, 0.14681, 0.67523, 0.64481],
        ["CNN Optimized", 0.07865, 0.19521, 0.74145, 0.64443]
    ]
    columns_dl = ["Model", "Hamming Loss", "F1-Score (Macro)", "F1-Score (Micro)", "Accuracy"]
    st.table(pd.DataFrame(data_dl, columns=columns_dl))

    st.markdown("<p style='text-align: center; font-style: italic;'>MultiTarget – Deep Learning Model Results</p>", unsafe_allow_html=True)

    st.header("Interpretability")
    st.subheader("Error Analysis")
    st.write("Since this is multi-label classification, we can generate label-wise confusion matrices rather than a standard one. For each label, we generated a confusion matrix to compare true vs. predicted values. Then, we visualized it using a heatmap to easily spot misclassifications.")

    # Figure 10
    image_dir = Path(__file__).parent / "pics"
    st.image(image_dir / "Figure 10.jpg", caption="Figure 10 MultiTarget - Confusion Matrix Heatmap")

    st.write("The confusion matrices reveal key insights into the model's performance across different labels. High-imbalance issues are evident in categories like Agricultural Supplies, Explosive Hazard Risk Awareness, and Mental Health Psychological Support, where very few positive cases were correctly predicted, indicating poor recall.")
    st.write("In contrast, labels like Food, Nutrition, WASH, and Cash Assistance Vouchers show better performance, with a higher number of true positives and relatively fewer false negatives.")
    st.write("However, some labels, such as Education and Livelihood Support, exhibit moderate false negatives, suggesting that the model struggles to correctly classify minority classes in certain categories.")
    st.write("Overall, the model performs well on frequently occurring labels but still struggles with rare ones, reinforcing the need for further optimization, such as threshold tuning, or alternative loss functions tailored for imbalanced multi-label classification.")

    st.subheader("Per-Class probability Threshold Optimization for XGBoost")

    # XGBoost Model Results - Table
    data_xgb = [
        ["XGBoost (0.001 feature selec.)", 0.01961, 0.58088, 0.84921, 0.78079],
        ["XGBoost (0.001+Per Class Threshold)", 0.019538, 0.63569, 0.85404, 0.779279]
    ]
    columns_xgb = ["Model", "Hamming Loss", "F1-Score (Macro)", "F1-Score (Micro)", "Accuracy"]
    st.table(pd.DataFrame(data_xgb, columns=columns_xgb))

    # Optimized Per-Class Thresholds - Table
    thresholds = {
        "Shelter": 0.4, "Health": 0.4, "NFIs": 0.3, "Electricity assistance": 0.3,
        "Food, nutrition": 0.5, "Agricultural supplies": 0.3, "Livelihood support": 0.3,
        "Education": 0.4, "WASH": 0.3, "Winterisation": 0.3, "Legal services": 0.3,
        "GBV services": 0.3, "CP services": 0.35, 
        "Explosive hazard risk awareness or removal of explosive contamination": 0.3,
        "Mental health psychological support": 0.3, "Cash assistance vouchers or cash in hand": 0.35
    }

    threshold_df = pd.DataFrame(thresholds.items(), columns=["Category", "Threshold"])
    st.table(threshold_df)


    st.subheader("SHAP analysis using XGBoost")

    # Figure 11
    image_dir = Path(__file__).parent / "pics"
    st.image(image_dir / "Figure 11.png", caption="Figure 11 MultiTarget - SHAP Values XGBoost")

    st.write("This SHAP summary plot illustrates the impact of the 340 most important features in the XGBoost model for Target Schelter. The left axis lists the top contributing features, where those at the top have the highest influence on the model’s predictions. The SHAP values on the x-axis indicate how much each feature pushes the prediction positively (right) or negatively (left). Red points represent high feature values, while blue points indicate low values.")

    st.subheader("SHAP Interpretation Across Labels")
    st.write("The SHAP analysis has provided critical insights into the drivers of humanitarian assistance predictions across 16 labels. These insights can help prioritize resources, target interventions more effectively, and build trust in the model's predictive capabilities through transparent, interpretable outputs.")

    # Table 18
    data = [
        ["Shelter Assistance", "Access to humanitarian aid, shelter damage, material shortages", 
        "Locations with severe shelter damage and resource limitations show increased need for assistance."],
        ["Health Assistance", "Barriers to healthcare, malnutrition treatment gaps", 
        "Lack of healthcare access strongly correlates with higher health assistance needs."],
        ["NFIs (Non-Food Items)", "Cooking fuel, shelter repair materials, coping strategies", 
        "Communities selling household items indicate a higher demand for NFIs."],
        ["Electricity Assistance", "Fuel costs, generator usage, power reliability", 
        "Areas with frequent power outages and high fuel expenses see increased predictions."],
        ["Food Assistance", "Food price inflation, meal skipping, reliance on credit", 
        "Food insecurity and high dependency on credit purchases predict higher food assistance needs."],
        ["Agricultural Supplies", "Food crop production levels, financial limitations", 
        "Low agricultural output correlates with increased need for agricultural assistance."],
        ["Livelihood Support", "Employment barriers, financial insecurity", 
        "Unemployment rates strongly influence livelihood support predictions."],
        ["Education Assistance", "School availability, child protection risks", 
        "Limited educational facilities and high early marriage risks drive education support needs."],
        ["WASH Assistance", "Water infrastructure issues, drinking water concerns", 
        "Poor water network conditions increase WASH assistance demands."],
        ["Winterization Assistance", "Heating deficiencies, seasonal conditions", 
        "Cold seasons and heating shortages predict higher winterization needs."],
        ["Legal Services Assistance", "Documentation challenges, protection risks", 
        "Households without legal documents show increased legal assistance requirements."],
        ["GBV Services Assistance", "Early marriage risks, protection concerns", 
        "Higher GBV risks in a region predict increased allocation of services."],
        ["CP Services Assistance", "Child labor prevalence, school dropout rates", 
        "Protection issues like child labor and dropout rates influence CP assistance needs."],
        ["Explosive Hazard Awareness Assistance", "Reported contamination risks, historical conflict data", 
        "Regions with known contamination show heightened awareness predictions."],
        ["Mental Health Assistance", "Psychological distress reports, lack of mental health services", 
        "Communities experiencing mental health crises predict increased support demand."],
        ["Cash Assistance", "Employment opportunities, financial barriers", 
        "Areas with severe financial insecurity predict higher cash assistance needs."]
    ]
    columns = ["Assistance Type", "Top Contributing Features", "SHAP Insights"]
    st.table(pd.DataFrame(data, columns=columns))
    st.markdown("<p style='text-align: center; font-style: italic;'>Table 18 <i>MultiTarget</i> - SHAP Interpretation Table</p>", unsafe_allow_html=True)

# 4. PREDICTION PAGE
elif selection == "Prediction":

    st.title("PREDICTION")

    st.write("""
    This page allows **live predictions** using multi-output classifier models. The process runs **in the background**, based on user-selected preferences, and may take a few minutes for some models.
    """)
    
    st.write("""
    Once you select a **model** and **location**, computation begins automatically. When ready, the **"Display Predictions"** button will appear. Results are shown only after clicking the button.
    """)
    st.write("""
    We use **5 models**, each predicting assistance needs **independently**. Since a camp may require **one or multiple types of aid**, we implemented a **MultiOutputClassifier**. You can select **one model** from the list below:
    """)

    # Model selection
    title = "### Select a Model"

    st.write("""
    - "best_xgb_340_features.pkl"
    - "best_xgb_model_with_thresholds.pkl" (**best performing model**)
    - "logistic_model.pkl", 
    - "svm_model_base.pkl"
    - "xgboost_model_optimized.joblib"
    
    **Note**: Some other models were trained but not used in production due to their performance are:
    
    - "random_forest_model.pkl"
    - "random_forest_optimized.pkl"
    - "svm_model_optimized.pkl"
    """)

    st.markdown(title)
    model_list = [
        "best_xgb_340_features.pkl", "best_xgb_model_with_thresholds.pkl", "logistic_model.pkl", 
        "svm_model_base.pkl", "xgboost_model_optimized.joblib"
    ]
    user_selected_model = st.selectbox("Choose a model:", model_list)

    #if st.button("Show Results"):
    # Define additional variables
    model_is_340 = user_selected_model in ["best_xgb_340_features.pkl", "best_xgb_model_with_thresholds.pkl"]
    model_is_all_features = user_selected_model not in ["best_xgb_340_features.pkl", "best_xgb_model_with_thresholds.pkl"]

    model_is_proba = user_selected_model in ["best_xgb_model_with_thresholds.pkl"]
    model_is_not_proba = user_selected_model not in ["best_xgb_model_with_thresholds.pkl"]

    # Load the filtering df 
    data_folder = Path(__file__).parent / "data and models" / "data"
    production_df_location_filtering = pd.read_csv(data_folder / "production_df_location_filtering.csv")

    # Define file paths
    model_folder = Path(__file__).parent / "data and models" / "models"
    data_folder = Path(__file__).parent / "data and models" / "data"

    # Load datasets conditionally
    if model_is_340:
        # X_train = pd.read_csv(os.path.join(data_folder, "X_train_340_features.csv"))
        # Y_train = pd.read_csv(os.path.join(data_folder, "Y_train_340_features.csv"))
        X_test = pd.read_csv(data_folder / "X_test_340_features.csv")
        Y_test = pd.read_csv(data_folder / "Y_test_340_features.csv")
        
    if model_is_all_features:
        # X_train = pd.read_csv(os.path.join(data_folder, "X_train_all_features.csv"))
        # Y_train = pd.read_csv(os.path.join(data_folder, "Y_train_all_features.csv"))
        X_test = pd.read_csv(data_folder / "X_test_all_features.csv")
        Y_test = pd.read_csv(data_folder / "Y_test_all_features.csv")
    
    # filter the filtering df based on exiting indices in the data
    production_df_location_filtering = production_df_location_filtering.loc[X_test.index]

    # Location selection
    st.markdown("### Select the location")

    st.write("""
    Locations are structured **hierarchically**:
    - **Governorate → Community → Camps**
    - Selecting a **governorate** filters available **communities**.
    - Selecting a **community** filters available **camps**.
    - **Districts and sub-districts** are excluded for simplicity.
    """)

    st.write("""
    Out of **16 Syrian governorates**, the dataset covers only **six**:
    **Al-Hasakeh, Aleppo, Ar-Raqqa, Deir-ez-Zor, Hama, and Idleb**.
    These regions fall under **Northwest Syria (NWS) and Northeast Syria (NES)**, which were opposition and Kurdish-controlled areas before December 2024.
    """)

    st.write("""
    After selecting a **governorate**, a **filtered list of communities** appears. You can choose **only one community**, each mapped to a **Locality (p-code system used by the UN)**.
    **Camps** serve as entry points for each locality.
    """)
    
    # Governorate selection
    governorates = sorted(production_df_location_filtering["Governorate"].unique())
    user_selected_governorate = st.selectbox("Choose a Governorate:", governorates)

    # Community selection
    filtered_df = production_df_location_filtering[
        production_df_location_filtering["Governorate"] == user_selected_governorate
    ]
    communities = sorted(filtered_df["Community"].unique())
    user_selected_community = st.selectbox("Choose a Community:", communities)

    # Camp ID selection
    try:
        filtered_camps = filtered_df[filtered_df["Community"] == user_selected_community]
        camp_ids = sorted(filtered_camps["camp_id"].unique())
        user_selected_camp_ids = st.multiselect("Select Camp:", camp_ids)

        # Obtain corresponding indices
        user_selected_camp_indices = production_df_location_filtering.index[production_df_location_filtering["camp_id"].isin(user_selected_camp_ids)].tolist()

        # Display selected values
        st.write("### Selected Options")
        """
        The values that you have selected are the following:
        """
        st.write(f"**Model:** {user_selected_model}")
        st.write(f"**Governorate:** {user_selected_governorate}")
        st.write(f"**Community:** {user_selected_community}")
        st.write(f"**Camp IDs:** {user_selected_camp_ids}")

        # Load the user_selected_model
        implemented_model = joblib.load(model_folder / user_selected_model)

        # if we are not predicting probabilities

        if model_is_not_proba:
            
            # Ensure feature alignment (important for correct predictions)
            X_test = X_test[implemented_model.feature_names_in_]  # Select the correct feature columns
            
            # Make predictions on the test set
            Y_pred = implemented_model.predict(X_test)
            
            # Convert predictions and actual values into a DataFrame with modified column names
            Y_pred = pd.DataFrame(Y_pred, index=X_test.index, columns=[f"{col}_Pred" for col in Y_test.columns])
            Y_test = pd.DataFrame(Y_test.values, index=Y_test.index, columns=[f"{col}_Actual" for col in Y_test.columns])

        # if we are not predicting probabilities only ("best_xgb_model_with_thresholds.pkl")

        if model_is_proba:
            # Extract thresholds and model from the loaded model
            thresholds = implemented_model.get("thresholds", None)
            implemented_model = implemented_model.get("model", None)
            
            # Convert the thresholds dictionary to a DataFrame
            thresholds_df = pd.DataFrame(list(thresholds.items()), columns=['Target', 'Threshold'])
            thresholds_df.set_index('Target', inplace=True)
            
            # Ensure feature alignment (important for correct predictions)
            X_test = X_test[implemented_model.feature_names_in_]
            
            # predict probabilities
            Y_prob = implemented_model.predict_proba(X_test)
            
            # We'll create a DataFrame where each column corresponds to a target's positive probabilities.
            target_names = list(thresholds.keys())
            positive_probs = {}

            for i, target in enumerate(target_names):
                # Extract the positive class probability (index 1) for each target
                positive_probs[target] = Y_prob[i][:, 1]
                
            # Create a DataFrame with shape (6583, 16)
            Y_prob = pd.DataFrame(positive_probs)
            
            Y_pred = Y_prob.copy()

            for target in Y_pred.columns:
                th = thresholds[target]
                # Set value to 1 if probability is above threshold, otherwise 0
                Y_pred[target] = (Y_pred[target] > th).astype(int)
            
            Y_pred.columns = [f"{col}_Pred" for col in Y_pred.columns]
            Y_test = pd.DataFrame(Y_test.values, index=Y_test.index, columns=[f"{col}_Actual" for col in Y_test.columns])

        # this interleaves the corresponding columns from Y_pred and Y_test
        frames = []
        for i in range(Y_pred.shape[1]):
            frames.append(pd.concat([Y_pred.iloc[:, i], Y_test.iloc[:, i]], axis=1))
        predictions_df = pd.concat(frames, axis=1)

        #these are the communities selected by the user
        displayed_communities = user_selected_camp_indices
        displayed_df = predictions_df.loc[displayed_communities]

        # Reshape the DataFrame
        melted_df = displayed_df.melt(ignore_index=False, var_name="Category", value_name="Value")

        # Extract category and type
        melted_df["Type"] = melted_df["Category"].apply(lambda x: "Pred" if "Pred" in x else "Actual")
        melted_df["Category"] = melted_df["Category"].str.replace("_Pred", "").str.replace("_Actual", "")

        # Pivot to required format
        predicted_assistance_df = melted_df.pivot_table(index=[melted_df.index, "Category"], columns="Type", values="Value").reset_index()

        # Rename columns
        predicted_assistance_df.columns.name = None  # Remove column names
        predicted_assistance_df.columns = ["Camp", "Aid assistance type needed", "Predicted values", "Actual values"]    
        # add camp_ to the name of community
        predicted_assistance_df["Camp"] = "camp_" + predicted_assistance_df["Camp"].astype(str)

        # fixing assistance name
        replacement_dict = {"Cash assistance vouchers or cash in hand": "Cash assistance", "Explosive hazard risk awareness or removal of explosive contamination": "Explosive removal", "Mental health psychological support": "Psychological support"}
        predicted_assistance_df["Aid assistance type needed"] = (predicted_assistance_df["Aid assistance type needed"].replace(replacement_dict, regex=True))

    # Display selected values
        st.write("### Show predictions")
        """
        So now go on and hit the botton below. You will be shown the live predicted values and the actual values for comparison!
        """

        # display the df
        if st.button("Display predictions"):
            st.dataframe(predicted_assistance_df)

    except ValueError:
        pass

# 5. PERSPECTIVES PAGE
elif selection == "Perspectives":
    st.title("PERSPECTIVES & TAKEAWAYS")

    # Summary of Achievements
    st.subheader("🔹 Key Achievements")
    st.write("""
    - Developed a high-performing **multi-label classification model** for humanitarian aid needs.
    - **XGBoost with per-class thresholding** outperformed other models.
    - **SHAP analysis** provided transparency on key decision factors.
    """)

    # Challenges Encountered
    st.subheader("⚠️ Challenges & Limitations")
    st.write("""
    - **Class Imbalance**: Some humanitarian aid types were underrepresented.
    - **High Dimensionality**: 497 features required significant preprocessing.
    - **Computational Costs**: Model tuning (SMOTE, thresholding, Grid Search) demanded high resources.
    """)

    # Future Improvements
    st.subheader("🚀 Future Directions")
    st.write("""
    - **Deep Learning Exploration**: Consider LSTMs or Transformer-based models.
    - **Advanced Oversampling**: Test **ADASYN** for improved minority class recall.
    - **Optimized Thresholding**: Use Bayesian Optimization or Genetic Algorithms.
    - **Feature Reduction**: Further streamline features to enhance model efficiency.
    """)

    # Real-World Impact
    st.subheader("🌍 Real-World Applications")
    st.write("""
    - **Aid Targeting**: Predicts humanitarian needs to improve response efforts.
    - **Resource Allocation**: Helps NGOs distribute aid more efficiently.
    - **Crisis Forecasting**: Provides insights for proactive planning in vulnerable areas.
    """)

    # Closing Statement
    st.markdown("**This project bridges ML and humanitarian efforts, offering an interpretable and scalable approach to aid distribution.**")

# 6. ABOUT PAGE
elif selection == "About":
    st.title("ABOUT")

    # Project Contributors
    st.subheader("👥 Project Team")
    st.write("""
    - **Ghiath Al Jebawi** - Data Scientist with 5 years of experience in humanitarian work in Syria, particularly in IDP camps.
    - **Bercin Ersoz** - Statistician specializing in data management, previously worked at the Turkish Central Bank.
    - **Allaeldene Ilou** - IT Project Manager focused on humanitarian and environmental projects.
    - **Caspar Stordeur** - Social scientist with expertise in data transformation, demographics, and community dynamics.
    """)

    # Data Sources
    st.subheader("📊 Data Sources")
    st.markdown("""
    - **[IMPACT REACH Initiative](https://www.impact-initiatives.org)**
    - **[HSOS Dataset & JMMI Dataset](https://www.impact-initiatives.org/resource-centre/?category%5b%5d=data_methods&location%5b%5d=231&programme%5b%5d=754&programme%5b%5d=764&programme%5b%5d=755&order=latest&limit=10)**
    """)

    # References
    st.subheader("📚 References")
    st.markdown("""
    - **Dong, H., Sun, J., & Sun, X. (2021).** A Multi-Objective Multi-Label Feature Selection Algorithm Based on Shapley Value. [Entropy, 23(8)](https://doi.org/10.3390/e23081094)
    - **scikit-learn.org (2025a).** [Precision, recall, and F-measures](https://scikit-learn/stable/modules/model_evaluation.html)
    - **scikit-learn.org (2025b).** [SGDClassifier](https://scikit-learn/stable/modules/generated/sklearn.linear_model.SGDClassifier.html)
    - **Shikun, Chen. (2021).** Interpretation of multi-label classification models using Shapley values. [arXiv](https://www.researchgate.net/publication/351046405_Interpretation_of_multi-label_classification_models_using_shapley_values)
    - **Sukhwani, N. (2020).** Handling Data Imbalance in Multi-label Classification (MLSMOTE). [TheCyPhy](https://medium.com/thecyphy/handling-data-imbalance-in-multi-label-classification-mlsmote-531155416b87)
    """)