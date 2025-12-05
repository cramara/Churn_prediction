import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import joblib


def preprocess(df):
    df = df.copy()

    # Handling rows with missing data
    st.write("Handling of missing data\n")

    # Counting rows before deletion
    nb_lignes_avant = len(df)
    st.write(f"Number of rows before processing : {nb_lignes_avant}")

    # Delete rows with too many missing values
    seuil_manquantes = 0.5  # 50% of the columns
    df = df[df.isnull().sum(axis=1) / len(df.columns) < seuil_manquantes]

    # Counting rows after deletion
    nb_lignes_apres = len(df)
    nb_lignes_supprimees = nb_lignes_avant - nb_lignes_apres

    st.write(f"Number of rows after deletion : {nb_lignes_apres}")
    st.write(f"Number of rows deleted : {nb_lignes_supprimees}")
    st.write(f"Percentage of data retained : {(nb_lignes_apres / nb_lignes_avant) * 100:.2f}%")

    # Check if there is any remaining missing data
    if df.isnull().sum().sum() > 0:
        st.write("\nRemaining missing data will be addressed in the cleaning section.")
        st.write(df.isnull().sum()[df.isnull().sum() > 0])
    else:
        st.write("\nAll missing data has been handled.")


    # Suppress useless columns
    df = df.drop(columns=["Unnamed: 0", "customerID"])

    # Convert TotalCharges into numerical values
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(0)  # New clients that haven't pay yet

    # Replace the NaNs in the services columns with "No"
    cols_with_na = ["OnlineBackup", "TechSupport", "StreamingTV", "StreamingMovies"]
    for col in cols_with_na:
        df[col] = df[col].fillna("No")

    # Standardize "No internet service" and "No phone service" 
    replace_map = {
        "No internet service": "No",
        "No phone service": "No"
    }   
    df = df.replace(replace_map)

    # Encode target (Churn) in binary
    df["Churn"] = df["Churn"].map({"No": 0, "Yes": 1})

    # Check the numeric/categorical columns
    categorical_cols = df.select_dtypes(include="object").columns
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
    st.write("Categorical columns:", list(categorical_cols))
    st.write("Numeric columns:", list(numeric_cols))

    # OneHot Encoding
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    return df


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

with st.expander("Data Preprocessing"):
    df_clean = preprocess(df)
    st.write("After cleaning : shape", df_clean.shape)

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

# Interactive Modeling
st.sidebar.header("Model Parameters")
test_size = st.sidebar.slider("Test Size", 0.1, 0.5, 0.3, 0.05)
C = st.sidebar.select_slider("C parameter (Inverse Regularization)", options=[0.01,0.1,1,10,100], value=1.0)
max_iter = st.sidebar.number_input("max_iter", min_value=100, max_value=10000, value=1000, step=100)

if st.sidebar.button("Train Model"):
    if "Churn" not in df_clean.columns:
        st.error("The 'Churn' column is missing after preprocessing.")
        st.stop()

    # Split train/test
    X = df_clean.drop("Churn", axis=1)
    y = df_clean["Churn"]
    feature_names = X.columns.tolist()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)

    # Standardisation
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Logistic Regression Model
    model = LogisticRegression(C=C, max_iter=int(max_iter))
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:,1]

    # Metrics
    st.subheader("Results")
    st.text(classification_report(y_test, y_pred, digits=3))
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
    plt.figure(figsize=(8,5))
    sns.barplot(x="nb_services", y="Churn", data=df_clean, ci=None, palette="viridis")
    plt.title("Churn rate based on the number of subscribed services", fontsize=14)
    plt.ylabel("Proportion of churn")
    plt.xlabel("Number of subscribed services")
    plt.show()

    # Check correlation
    corr_nb_services = df_clean["nb_services"].corr(df_clean["Churn"])
    st.write(f"Correlation nb_services vs churn : {corr_nb_services:.3f}")

    # Detailed churn rate by number of services
    churn_by_services = df_clean.groupby("nb_services")["Churn"].mean()
    st.write("\nChurn rate by number of services :\n", churn_by_services)


    # save model + scaler + features
    if st.button("Save the model"):
        joblib.dump({"model": model, "scaler": scaler, "features": feature_names}, "churn_model.joblib")
        st.success("Model saved : churn_model.joblib")

# Exploration per filter
st.sidebar.header("Data Exploration")
if st.sidebar.checkbox("Display Filterable Table"):
    st.write(df_clean.head(200))

st.sidebar.markdown("### Export")
if st.sidebar.button("Download Cleaned Data"):
    csv = df_clean.to_csv(index=False).encode('utf-8')
    st.download_button("Download CSV", data=csv, file_name="data_churn_clean.csv", mime="text/csv")
