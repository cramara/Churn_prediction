import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve
import joblib


def preprocess(df, verbose=True):
    df = df.copy()

    # Handling rows with missing data
    if verbose:
        st.write("Handling of missing data\n")

    # Counting rows before deletion
    nb_lignes_avant = len(df)
    if verbose:
        st.write(f"Number of rows before processing : {nb_lignes_avant}")

    # Delete rows with too many missing values
    seuil_manquantes = 0.5  # 50% of the columns
    df = df[df.isnull().sum(axis=1) / len(df.columns) < seuil_manquantes]

    # Counting rows after deletion
    nb_lignes_apres = len(df)
    nb_lignes_supprimees = nb_lignes_avant - nb_lignes_apres

    if verbose:
        st.write(f"Number of rows after deletion : {nb_lignes_apres}")
        st.write(f"Number of rows deleted : {nb_lignes_supprimees}")
        st.write(f"Percentage of data retained : {(nb_lignes_apres / nb_lignes_avant) * 100:.2f}%")

    # Check if there is any remaining missing data
    if verbose:
        if df.isnull().sum().sum() > 0:
            st.write("\nRemaining missing data will be addressed in the cleaning section.")
            st.write(df.isnull().sum()[df.isnull().sum() > 0])
        else:
            st.write("\nAll missing data has been handled.")

    # Suppress useless columns (handle errors if columns don't exist)
    cols_to_drop = []
    if "Unnamed: 0" in df.columns:
        cols_to_drop.append("Unnamed: 0")
    if "customerID" in df.columns:
        cols_to_drop.append("customerID")
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)

    # Convert TotalCharges into numerical values
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        df["TotalCharges"] = df["TotalCharges"].fillna(0)  # New clients that haven't pay yet

    # Replace the NaNs in the services columns with "No"
    cols_with_na = ["OnlineBackup", "TechSupport", "StreamingTV", "StreamingMovies"]
    for col in cols_with_na:
        if col in df.columns:
            df[col] = df[col].fillna("No")

    # Standardize "No internet service" and "No phone service" 
    replace_map = {
        "No internet service": "No",
        "No phone service": "No"
    }   
    df = df.replace(replace_map)

    # Encode target (Churn) in binary
    if "Churn" in df.columns:
        df["Churn"] = df["Churn"].map({"No": 0, "Yes": 1})

    # Check the numerical/categorical columns
    if verbose:
        categorical_cols = df.select_dtypes(include="object").columns
        numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
        st.write("Colonnes catégorielles :", list(categorical_cols))
        st.write("Colonnes numériques :", list(numeric_cols))

    # OneHot Encoding
    categorical_cols = df.select_dtypes(include="object").columns
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    return df


def train_model(df_clean, test_size=0.3, C=1.0, max_iter=1000):
    """Train the model automatically on the cleaned data"""
    if "Churn" not in df_clean.columns:
        return None, None, None
    
    # Split train/test
    X = df_clean.drop("Churn", axis=1)
    y = df_clean["Churn"]
    feature_names = X.columns.tolist()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
    
    # Standardisation
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Logistic Regression Model
    model = LogisticRegression(C=C, max_iter=int(max_iter))
    model.fit(X_train_scaled, y_train)
    
    return model, scaler, feature_names, X_test_scaled, y_test


st.set_page_config(layout="wide", page_title="Churn Analysis")
st.title("Dashboard Churn Prediction")
st.markdown("""
This single-page application :
    - Upload the dataset  
    - Preprocess the data  
    - Explore the dataset visually  
    - Train a machine learning model  
""")

# Upload or Default local reading
uploaded = st.file_uploader("Upload CSV (data_churn.csv)", type=["csv"])
if uploaded is not None:
    df = pd.read_csv(uploaded)
else:
    st.info("No file uploaded, the dashboard attempts to load data_churn.csv locally.")
    try:
        df = pd.read_csv("data_churn.csv")
    except Exception as e:
        st.error("Unable to load 'data_churn.csv'. Upload a CSV file.")
        st.stop()

# Data overview
with st.expander("Data Overview"):
    st.write("Shape:", df.shape)
    st.dataframe(df.head())

# Missing values
with st.expander("Missing Values Analysis"):
    missing_data = pd.DataFrame({
        'Number': df.isnull().sum(), 
        'Percentage': df.isnull().sum()/len(df)*100})
    missing_data = missing_data[missing_data['Number']>0].sort_values('Number', ascending=False)
    if missing_data.shape[0] == 0:
        st.success("No missing values detected.")
    else:
        st.dataframe(missing_data)
        fig, ax = plt.subplots(figsize=(8,4))
        sns.heatmap(df.isnull(), yticklabels=False, cbar=True, cmap='viridis')
        ax.set_title("Missing Values Heatmap", fontsize=14)
        st.pyplot(fig)

# Detect if CSV is for training (has Churn column) or prediction (no Churn column)
has_churn_column = "Churn" in df.columns

# Model Parameters in Sidebar (only shown if training mode)
if has_churn_column:
    st.sidebar.header("Model Parameters")
    test_size = st.sidebar.slider("Test Size", 0.1, 0.5, 0.3, 0.05)
    C = st.sidebar.select_slider("C parameter (Inverse Regularization)", options=[0.01,0.1,1,10,100], value=1.0)
    max_iter = st.sidebar.number_input("max_iter", min_value=100, max_value=10000, value=1000, step=100)
else:
    # Use default parameters for loading model
    test_size = 0.3
    C = 1.0
    max_iter = 1000

# Preprocess data automatically (silent mode)
df_clean = preprocess(df, verbose=False)

# Store df_clean in session state for use in other sections
st.session_state["df_clean"] = df_clean
st.session_state["has_churn_column"] = has_churn_column
st.session_state["df_original"] = df  # Store original for predictions

# Automatic detection: Training mode or Prediction mode
if has_churn_column:
    # TRAINING MODE: CSV contains Churn column, train the model
    st.info("📊 **Training Mode**: CSV contains 'Churn' column. Model will be trained automatically.")
    
    # Check if we need to retrain (if parameters changed or model doesn't exist)
    need_retrain = (
        "model" not in st.session_state or 
        st.session_state.get("test_size") != test_size or
        st.session_state.get("C") != C or
        st.session_state.get("max_iter") != max_iter or
        st.session_state.get("has_churn_column") != has_churn_column
    )
    
    if need_retrain:
        with st.spinner("Training model automatically..."):
            model, scaler, feature_names, X_test_scaled, y_test = train_model(df_clean, test_size=test_size, C=C, max_iter=max_iter)
            
            if model is not None:
                # Store in session state
                st.session_state["model"] = model
                st.session_state["scaler"] = scaler
                st.session_state["feature_names"] = feature_names
                st.session_state["X_test_scaled"] = X_test_scaled
                st.session_state["y_test"] = y_test
                st.session_state["test_size"] = test_size
                st.session_state["C"] = C
                st.session_state["max_iter"] = max_iter
                
                # Automatically save the model
                try:
                    joblib.dump({"model": model, "scaler": scaler, "features": feature_names}, "churn_model.joblib")
                    st.success("✅ Model trained and saved automatically: churn_model.joblib")
                except Exception as e:
                    st.warning(f"Model trained but could not be saved: {e}")
else:
    # PREDICTION MODE: CSV doesn't contain Churn column, load saved model
    st.info("🔮 **Prediction Mode**: CSV doesn't contain 'Churn' column. Loading saved model for predictions.")
    
    # Try to load model from session state first, then from file
    if "model" in st.session_state and st.session_state["model"] is not None:
        st.success("✅ Model loaded from session state.")
    else:
        try:
            model_data = joblib.load("churn_model.joblib")
            model = model_data["model"]
            scaler = model_data["scaler"]
            feature_names = model_data["features"]
            
            # Store in session state
            st.session_state["model"] = model
            st.session_state["scaler"] = scaler
            st.session_state["feature_names"] = feature_names
            st.success("✅ Model loaded from file: churn_model.joblib")
        except FileNotFoundError:
            st.error("❌ No saved model found (churn_model.joblib). Please upload a CSV with 'Churn' column first to train the model.")
        except Exception as e:
            st.error(f"❌ Error loading model: {e}")

with st.expander("Data Preprocessing"):
    # Show verbose preprocessing details
    df_clean_display = preprocess(df, verbose=True)
    st.write("After cleaning : shape", df_clean_display.shape)

# EDA section only shown in training mode (when Churn column exists)
if has_churn_column:
    with st.expander("EDA"):
        col1, col2 = st.columns([1,1])
        with col1:
            if "Churn" in df_clean.columns:
                st.subheader("Churn Distribution")
                fig, ax = plt.subplots(figsize=(4,3))
                sns.countplot(x='Churn', data=df_clean, ax=ax)
                st.pyplot(fig)
                st.write(df_clean['Churn'].value_counts(normalize=True).mul(100).round(2))
        with col2:
            if "MonthlyCharges" in df_clean.columns and "Churn" in df_clean.columns:
                st.subheader("MonthlyCharges vs Churn")
                fig, ax = plt.subplots(figsize=(5,3))
                sns.boxplot(x='Churn', y='MonthlyCharges', data=df_clean, ax=ax)
                st.pyplot(fig)

        # Correlation/heatmap
        if st.checkbox("Display Correlation Matrix"):
            corr = df_clean.corr()
            fig, ax = plt.subplots(figsize=(10,8))
            sns.heatmap(corr, cmap="coolwarm", center=0, annot=False)
            st.pyplot(fig)
            if "Churn" in corr.columns:
                st.write("Top Correlations with Churn :")
                st.dataframe(corr["Churn"].sort_values(ascending=False).head(10))

with st.expander("Prediction"):
    st.subheader("Churn Prediction")
    
    # Check if model exists in session state (trained automatically) or loaded from file
    if "model" in st.session_state and st.session_state["model"] is not None:
        model = st.session_state["model"]
        scaler = st.session_state["scaler"]
        feature_names = st.session_state["feature_names"]
        st.success("Model ready for predictions!")
        
        # If in prediction mode (no Churn column), automatically use uploaded CSV for batch prediction
        if not has_churn_column:
            # Auto-predict mode: use the uploaded CSV automatically
            st.info("📊 **Auto-Prediction Mode**: Using uploaded CSV for batch predictions.")
            pred_df = df.copy()  # Use the uploaded CSV
            
            if st.button("Predict Churn for All Customers in Uploaded CSV"):
                with st.spinner("Processing predictions..."):
                    # Preprocess prediction data
                    pred_df_processed = pred_df.copy()
                    
                    # Drop customerID if present
                    if "customerID" in pred_df_processed.columns:
                        pred_df_processed = pred_df_processed.drop(columns=["customerID"])
                    if "Unnamed: 0" in pred_df_processed.columns:
                        pred_df_processed = pred_df_processed.drop(columns=["Unnamed: 0"])
                    
                    # Convert TotalCharges
                    if "TotalCharges" in pred_df_processed.columns:
                        pred_df_processed["TotalCharges"] = pd.to_numeric(pred_df_processed["TotalCharges"], errors="coerce")
                        pred_df_processed["TotalCharges"] = pred_df_processed["TotalCharges"].fillna(0)
                    
                    # Fill missing values in service columns
                    cols_with_na = ["OnlineBackup", "TechSupport", "StreamingTV", "StreamingMovies"]
                    for col in cols_with_na:
                        if col in pred_df_processed.columns:
                            pred_df_processed[col] = pred_df_processed[col].fillna("No")
                    
                    # Standardize
                    pred_df_processed = pred_df_processed.replace({
                        "No internet service": "No",
                        "No phone service": "No"
                    })
                    
                    # OneHot Encoding
                    categorical_cols = pred_df_processed.select_dtypes(include="object").columns
                    pred_df_processed = pd.get_dummies(pred_df_processed, columns=categorical_cols, drop_first=True)
                    
                    # Ensure all feature columns are present
                    for col in feature_names:
                        if col not in pred_df_processed.columns:
                            pred_df_processed[col] = 0
                    
                    # Reorder columns
                    pred_df_processed = pred_df_processed[feature_names]
                    
                    # Scale
                    X_pred = scaler.transform(pred_df_processed)
                    
                    # Predictions
                    predictions = model.predict(X_pred)
                    probabilities = model.predict_proba(X_pred)[:, 1]
                    
                    # Create results dataframe
                    results_df = pred_df.copy()
                    results_df["Predicted_Churn"] = ["Yes" if p == 1 else "No" for p in predictions]
                    results_df["Churn_Probability"] = probabilities
                    
                    st.subheader("Prediction Results")
                    st.dataframe(results_df)
                    
                    # Summary statistics
                    st.subheader("Summary")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Customers", len(results_df))
                    with col2:
                        st.metric("Predicted Churn", sum(predictions))
                    with col3:
                        st.metric("Churn Rate", f"{sum(predictions)/len(predictions):.2%}")
                    
                    # Download results
                    csv_results = results_df.to_csv(index=False).encode('utf-8')
                    st.download_button("Download Predictions CSV", data=csv_results, file_name="churn_predictions.csv", mime="text/csv")
        else:
            # Training mode: allow both single customer and batch CSV upload
            # Use the model for predictions
            prediction_mode = st.radio("Prediction Mode", ["Single Customer", "Batch (CSV)"], horizontal=True)
            
            if prediction_mode == "Single Customer":
                st.markdown("### Predict Churn for a Single Customer")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    gender = st.selectbox("Gender", ["Male", "Female"])
                    senior_citizen = st.selectbox("Senior Citizen", [0, 1])
                    partner = st.selectbox("Partner", ["Yes", "No"])
                    dependents = st.selectbox("Dependents", ["Yes", "No"])
                    tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, value=12)
                    phone_service = st.selectbox("Phone Service", ["Yes", "No"])
                    multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
                    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
                    
                with col2:
                    online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
                    online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
                    device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
                    tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
                    streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
                    streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
                    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
                    paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
                    payment_method = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
                
                monthly_charges = st.number_input("Monthly Charges", min_value=0.0, max_value=200.0, value=50.0, step=0.1)
                total_charges = st.number_input("Total Charges", min_value=0.0, value=500.0, step=0.1)
                
                if st.button("Predict Churn"):
                    # Create a dictionary with customer data
                    customer_data = {
                        "gender": gender,
                        "SeniorCitizen": senior_citizen,
                        "Partner": partner,
                        "Dependents": dependents,
                        "tenure": tenure,
                        "PhoneService": phone_service,
                        "MultipleLines": multiple_lines,
                        "InternetService": internet_service,
                        "OnlineSecurity": online_security,
                        "OnlineBackup": online_backup,
                        "DeviceProtection": device_protection,
                        "TechSupport": tech_support,
                        "StreamingTV": streaming_tv,
                        "StreamingMovies": streaming_movies,
                        "Contract": contract,
                        "PaperlessBilling": paperless_billing,
                        "PaymentMethod": payment_method,
                        "MonthlyCharges": monthly_charges,
                        "TotalCharges": total_charges
                    }
                    
                    # Convert to DataFrame
                    customer_df = pd.DataFrame([customer_data])
                    
                    # Preprocess the data (similar to training preprocessing)
                    customer_df_processed = customer_df.copy()
                    
                    # Standardize "No internet service" and "No phone service"
                    customer_df_processed = customer_df_processed.replace({
                        "No internet service": "No",
                        "No phone service": "No"
                    })
                    
                    # OneHot Encoding
                    categorical_cols = customer_df_processed.select_dtypes(include="object").columns
                    customer_df_processed = pd.get_dummies(customer_df_processed, columns=categorical_cols, drop_first=True)
                    
                    # Ensure all feature columns from training are present
                    for col in feature_names:
                        if col not in customer_df_processed.columns:
                            customer_df_processed[col] = 0
                    
                    # Reorder columns to match training features
                    customer_df_processed = customer_df_processed[feature_names]
                    
                    # Scale the features
                    X_customer = scaler.transform(customer_df_processed)
                    
                    # Make prediction
                    prediction = model.predict(X_customer)[0]
                    probability = model.predict_proba(X_customer)[0]
                    
                    # Display results
                    st.markdown("### Prediction Results")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if prediction == 1:
                            st.error(f"**Prediction: CHURN** (Probability: {probability[1]:.2%})")
                        else:
                            st.success(f"**Prediction: NO CHURN** (Probability: {probability[0]:.2%})")
                    
                    with col2:
                        st.metric("Churn Probability", f"{probability[1]:.2%}")
                        st.metric("No Churn Probability", f"{probability[0]:.2%}")
                    
                    # Progress bar
                    st.progress(probability[1])
                    st.caption(f"Churn Risk: {probability[1]:.2%}")
            
            else:  # Batch prediction
                st.markdown("### Batch Prediction from CSV")
                st.info("Upload a CSV file with customer data (same format as training data)")
                
                uploaded_pred = st.file_uploader("Upload CSV for prediction", type=["csv"], key="prediction_csv")
                
                if uploaded_pred is not None:
                    pred_df = pd.read_csv(uploaded_pred)
                    st.write("Preview of uploaded data:")
                    st.dataframe(pred_df.head())
                    
                    if st.button("Predict Churn for All Customers"):
                        # Preprocess prediction data
                        pred_df_processed = pred_df.copy()
                        
                        # Drop customerID if present
                        if "customerID" in pred_df_processed.columns:
                            pred_df_processed = pred_df_processed.drop(columns=["customerID"])
                        if "Unnamed: 0" in pred_df_processed.columns:
                            pred_df_processed = pred_df_processed.drop(columns=["Unnamed: 0"])
                        
                        # Convert TotalCharges
                        pred_df_processed["TotalCharges"] = pd.to_numeric(pred_df_processed["TotalCharges"], errors="coerce")
                        pred_df_processed["TotalCharges"] = pred_df_processed["TotalCharges"].fillna(0)
                        
                        # Fill missing values in service columns
                        cols_with_na = ["OnlineBackup", "TechSupport", "StreamingTV", "StreamingMovies"]
                        for col in cols_with_na:
                            if col in pred_df_processed.columns:
                                pred_df_processed[col] = pred_df_processed[col].fillna("No")
                        
                        # Standardize
                        pred_df_processed = pred_df_processed.replace({
                            "No internet service": "No",
                            "No phone service": "No"
                        })
                        
                        # OneHot Encoding
                        categorical_cols = pred_df_processed.select_dtypes(include="object").columns
                        pred_df_processed = pd.get_dummies(pred_df_processed, columns=categorical_cols, drop_first=True)
                        
                        # Ensure all feature columns are present
                        for col in feature_names:
                            if col not in pred_df_processed.columns:
                                pred_df_processed[col] = 0
                        
                        # Reorder columns
                        pred_df_processed = pred_df_processed[feature_names]
                        
                        # Scale
                        X_pred = scaler.transform(pred_df_processed)
                        
                        # Predictions
                        predictions = model.predict(X_pred)
                        probabilities = model.predict_proba(X_pred)[:, 1]
                        
                        # Create results dataframe
                        results_df = pred_df.copy()
                        results_df["Predicted_Churn"] = ["Yes" if p == 1 else "No" for p in predictions]
                        results_df["Churn_Probability"] = probabilities
                        
                        st.subheader("Prediction Results")
                        st.dataframe(results_df)
                        
                        # Summary statistics
                        st.subheader("Summary")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total Customers", len(results_df))
                        with col2:
                            st.metric("Predicted Churn", sum(predictions))
                        with col3:
                            st.metric("Churn Rate", f"{sum(predictions)/len(predictions):.2%}")
                        
                        # Download results
                        csv_results = results_df.to_csv(index=False).encode('utf-8')
                        st.download_button("Download Predictions CSV", data=csv_results, file_name="churn_predictions.csv", mime="text/csv")
    else:
        st.info("Please upload and preprocess data first. The model will be trained automatically after preprocessing.")

# Model Evaluation (display results of automatically trained model) - only in training mode
if has_churn_column:
    with st.expander("Model Evaluation"):
        if "model" in st.session_state and st.session_state["model"] is not None and "X_test_scaled" in st.session_state:
            model = st.session_state["model"]
            scaler = st.session_state["scaler"]
            feature_names = st.session_state["feature_names"]
            X_test_scaled = st.session_state["X_test_scaled"]
            y_test = st.session_state["y_test"]
            df_clean = st.session_state["df_clean"]
            
            # Make predictions
            y_pred = model.predict(X_test_scaled)
            y_proba = model.predict_proba(X_test_scaled)[:,1]
            
            # Metrics
            st.subheader("Model Performance")
            st.code(classification_report(y_test, y_pred, digits=3))
            auc = roc_auc_score(y_test, y_proba)
            st.write(f"ROC AUC : {auc:.3f}")
            
            # Confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots(figsize=(4,3))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
            ax.set_title("Confusion matrix")
            ax.set_xlabel("Predicted label")
            ax.set_ylabel("True label")
            plt.xticks([0.5, 1.5], ["Non churn", "Churn"])
            plt.yticks([0.5, 1.5], ["Non churn", "Churn"], rotation=0)
            st.pyplot(fig)
            
            # ROC curve
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            fig, ax = plt.subplots()
            ax.plot(fpr, tpr)
            ax.plot([0,1],[0,1], linestyle="--")
            ax.set_title("ROC curve")
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            st.pyplot(fig)
            
            # Feature importance (coeff)
            coef = pd.DataFrame({"feature": feature_names, "coef": model.coef_[0]})
            coef['abs_coef'] = coef['coef'].abs()
            #Select the 10 largest in absolute value
            top_features = coef.sort_values("abs_coef", ascending=False).head(10)
            colors = ["tomato" if c > 0 else "steelblue" for c in top_features["coef"]]
            fig, ax = plt.subplots(figsize=(6,4))
            bars = ax.barh(top_features['feature'], top_features['coef'], color=colors)
            ax.axvline(0, color='black', linewidth=0.8)
            st.subheader("Top features (coefficients)")
            ax.set_xlabel("Coefficient")
            ax.set_ylabel("Variable")
            st.dataframe(top_features)
            for bar in bars:
                ax.text(bar.get_width(), bar.get_y() + bar.get_height()/2,
                        f"{bar.get_width():.2f}", va='center',
                        ha="left" if bar.get_width() > 0 else "right")
            st.pyplot(fig)
            
            # List of columns corresponding to subscribed services
            service_cols = [
                "PhoneService_Yes",
                "MultipleLines_Yes",
                "InternetService_Fiber optic",
                "OnlineSecurity_Yes",
                "OnlineBackup_Yes",
                "DeviceProtection_Yes",
                "TechSupport_Yes",
                "StreamingTV_Yes",
                "StreamingMovies_Yes"
            ]
            
            # Create a variable "nb_services"
            df_clean["nb_services"] = df_clean[service_cols].sum(axis=1)
            
            # Analysis : churn based on the number of services
            fig, ax = plt.subplots(figsize=(8,5))
            sns.barplot(x="nb_services", y="Churn", data=df_clean, ci=None, palette="viridis", ax=ax)
            ax.set_title("Churn rate based on the number of subscribed services", fontsize=14)
            ax.set_ylabel("Proportion of churn")
            ax.set_xlabel("Number of subscribed services")
            st.pyplot(fig)
            
            # Check correlation
            corr_nb_services = df_clean["nb_services"].corr(df_clean["Churn"])
            st.write(f"Correlation nb_services vs churn : {corr_nb_services:.3f}")
            
            # Detailed churn rate by number of services
            churn_by_services = df_clean.groupby("nb_services")["Churn"].mean()
            st.write("\nChurn rate by number of services :\n", churn_by_services)
            
            # Save model button (already saved automatically, but allow manual save)
            if st.button("Save the model"):
                joblib.dump({"model": model, "scaler": scaler, "features": feature_names}, "churn_model.joblib")
                st.success("Model saved : churn_model.joblib")
        else:
            st.info("Model will be trained automatically after data preprocessing.")

# Exploration per filter
st.sidebar.header("Data Exploration")
if st.sidebar.checkbox("Display Filterable Table"):
    df_clean_display = st.session_state.get("df_clean", None)
    if df_clean_display is not None:
        st.write(df_clean_display.head(200))
    else:
        st.info("Please preprocess data first.")

st.sidebar.markdown("### Export")
if st.sidebar.button("Download Cleaned Data"):
    df_clean_display = st.session_state.get("df_clean", None)
    if df_clean_display is not None:
        csv = df_clean_display.to_csv(index=False).encode('utf-8')
        st.download_button("Download CSV", data=csv, file_name="data_churn_clean.csv", mime="text/csv")
    else:
        st.warning("No cleaned data available. Please preprocess data first.")
