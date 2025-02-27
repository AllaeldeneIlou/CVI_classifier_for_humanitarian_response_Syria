import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

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
    
    # Define a single tab
    tabs = st.tabs(["HSOS Indicators"])

    with tabs[0]:
        # Centered Content with Custom Styling
        st.markdown("""
            <div style="text-align: center;">
                <h3>The <b>HSOS dataset</b> contains monthly reports from <b>2019 to 2024</b>, detailing information across <b>11 indicator groups</b>:</h3>
            </div>
        """, unsafe_allow_html=True)
        
        # Centered List in Markdown with a fixed-width box
        st.markdown("""
            <div style="display: flex; justify-content: center;">
                <div style="text-align: left; background-color: #f8f9fa; padding: 15px; border-radius: 10px; width: 50%;">
                    <ol>
                        <li>Demographics</li>
                        <li>Shelter</li>
                        <li>Electricity & Non-Food Items (NFIs)</li>
                        <li>Food Security</li>
                        <li>Livelihoods</li>
                        <li>Water, Sanitation, and Hygiene (WASH)</li>
                        <li>Health</li>
                        <li>Education</li>
                        <li>Protection</li>
                        <li>Accountability & Humanitarian Assistance</li>
                        <li>Priority Needs</li>
                    </ol>
                </div>
            </div>
        """, unsafe_allow_html=True)

    # Additional Description Below
    st.write("""    
    A total of **110 HSOS files** were initially analyzed. To maintain **data consistency**, the dataset was limited to reports from **2021 to 2023**, reducing the selection to **61 files**. Since no standardized catalog of questions exists across files, an **indicator selection process** was applied, retaining only those present in at least **57 of the 61 files**. The final dataset consists of **1,668 indicators** and **52,095 observations**.
    
    The **JMMI dataset**, focusing on the **Survival Minimum Expenditure Basket (SMEB)**, complements the HSOS data by analyzing economic conditions and market trends. The dataset originally contained **123 columns**, detailing item-specific prices. To streamline the analysis, individual item prices were removed in favor of **aggregated** subtotals, totals, and categorical variables.
    
    After restricting data to the years **2021 to 2023**, the final **JMMI dataset** includes **21 indicators** and over **4,700 observations**, ensuring alignment with the HSOS dataset.
    """)

# 2. EDA PAGE
elif selection == "EDA":
    st.title("Exploratory Data Analysis")

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
        col1, col2 = st.columns([1, 2], gap='large')
        with col1:
            st.image(image_dir / "4_Demographic_distribution_across_governorate.png", caption="Demographic distribution across governorates", width=int(0.5 * 800))
        with col2:
            st.markdown(
            "This **first pie chart illustrates the percentage distribution of the assessed population across governorates.** "
            "Idleb and Aleppo dominate, with similar proportions (**31.3%** and **30.8%**, respectively), while **Deir-ez-Zor** and other smaller regions contribute minimally to the dataset."
        )

        col3, col4 = st.columns([1, 1], gap='large')
        with col3:
            st.image(image_dir / "5_Percentage_distribution_districts.png", caption="Percentage distribution across districts", width=int(0.8 * 800))
        with col4:
            st.markdown(
            "This **second pie chart shows the distribution across districts.** "
            "A significant proportion (**36.1%**) is categorized as **'Other'** (sum of categories below 5%), followed by **Ar-Raqqa (10.9%)** and **Harim (9.7%)**. "
            "This highlights the geographic spread and variability in representation at the district level, which includes **23 unique values**."
        )

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
    st.title("Model Performance")
    st.write("🚧 Modelling functionality will be implemented soon.")

# 4. PREDICTION PAGE
elif selection == "Prediction":
    st.title("Make a Prediction")
    st.write("🚧 Prediction functionality will be implemented soon.")

# 5. PERSPECTIVES PAGE
elif selection == "Perspectives":
    st.title("Perspectives and Key Takeaways")

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
    st.title("About the Project")

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
