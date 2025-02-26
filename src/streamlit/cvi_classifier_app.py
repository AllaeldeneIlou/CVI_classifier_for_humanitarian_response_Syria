# 3. MODELING PAGE
elif selection == "Modeling":

    import streamlit as st
    import pandas as pd

    # Title 1 (Biggest)
    st.title("MODELING")

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
    st.write("XGBoost (Per-Class Thresholds) Results:")
    st.write("Hamming Loss: 0.03136089388977124")
    st.write("F1-Score (Macro): 0.6365977808806301")
    st.write("F1-Score (Micro): 0.802532913146984")
    st.write("Accuracy: 0.7796532462579")
    st.write("Optimized Per-Class Thresholds:")
    st.write("""[
    "Shelter": 0.29999999999999993,
    "Health": 0.30000000000000004,
    "NFI's": 0.3,
    "Electricity assistance": 0.3,
    "Food, nutrition": 0.49999999999999994,
    "Agricultural supplies": 0.3,
    "Livelihood support": 0.3,
    "Wash": 0.3,
    "Winterization": 0.3,
    "Legal services": 0.3,
    "GBV services": 0.3,
    "CP services": 0.35,
    "Explosive hazard risk awareness or removal of explosive contamination": 0.3,
    "Mental health psychological support": 0.3,
    "Cash assistance vouchers_pre cash in hand": 0.35
    ]""")
    st.write("The per-class threshold tuning has notably improved performance, especially for macro F1-score and micro F1-score, while maintaining accuracy at a similar level. Some common labels (e.g., Food, nutrition, Cash assistance, CP services) have higher thresholds (0.35 - 0.5). This prevents over-prediction and reduces false positives.")
    st.write("Some minority labels (e.g., GBV services, Electricity assistance, NFIs, WASH) have lower thresholds (0.3 - 0.35). This improves recall, helping to detect rare events better.")

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

    import streamlit as st
    import joblib
    import pandas as pd
    from sklearn.model_selection import train_test_split
    import os
    import numpy as np

    st.title("PREDICTION")

    """
    This page is dedicated for performing live predictions using the multi-output classifier models.
    The prediction process runs actually in the background based on the user's selected preferences, therefore, the waiting time might take up to a few minutes for some models.
    """
    """
    Once you select the model and the location preferences, the compute will start, and the and the "Display predictions" botton will appear.
    The results will be shown only after the user hits the botton "Display predictions".
    """

    # Model selection
    title = "### Select a Model"

    """
    We are working with 8 models that predit the classes separately. So a camp can need any any class or combination of classes, therefore we used the MultiOutputClassifier.
    You will be able to select a single model out of the following list:

    - "best_xgb_340_features.pkl"
    - "best_xgb_model_with_thresholds.pkl" (best performing model)
    - "logistic_model.pkl", 
    - "svm_model_base.pkl"
    - "xgboost_model_optimized.joblib"
    note: some models are functional but couldn't be put in production because their individual sizes exceed 100mb. More specifically they are:
    - "random_forest_model.pkl"
    - "random_forest_optimized.pkl"
    - "svm_model_optimized.pkl"
    """

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
    """
    The location attributes (Governorate, Community, and Camps) are organized hierarchially in one direction.
    So The choice of a governorate will narrow down the choices of community to include only communities that are within this governorate. Same goes for camps.
    We have skipped intermediary location attributes such as district and sub-district to avoid complication.
    """
    """
    Syria has 16 governorates. The available data coveres only six governorates, which are: Al-Hasakeh, Aleppo, Ar-Raqqa, Deir-ez-Zor, Hama, and Idleb.
    The areas of these governorates exist under the areas of Northwest Syria (NWS) and Northeast Syria (NES) which were the geopolitical regions hosting opposition and Kurdish controlled areas before December 2024.
    """
    """
    Once the governorate is selected a nawrrowed down list of the available communities will be within the dropdown list. Only one community could be selected. a Community correspond to a Locality with the p-code used in the UN coding for localities in Syria
    Camps are the entrypoints in each locality.
    """
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