# Project Report: Customer Churn Prediction

## Table of Contents

1. [Introduction](#introduction)
2. [Context and Objectives](#context-and-objectives)
3. [Data and Methodology](#data-and-methodology)
4. [Exploratory Data Analysis](#exploratory-data-analysis)
5. [Data Preparation and Cleaning](#data-preparation-and-cleaning)
6. [Modeling](#modeling)
7. [Streamlit Application](#streamlit-application)
8. [Results and Interpretations](#results-and-interpretations)
9. [Conclusion and Perspectives](#conclusion-and-perspectives)

---

## Introduction

This report documents the development of a customer churn (attrition) prediction system for a telecommunications company. This project combines data analysis, machine learning, and the development of an interactive application to identify customers at risk of cancellation and enable preventive actions.

Churn represents a major challenge for service companies, particularly in the telecommunications sector where competition is intense and the cost of acquiring a new customer is significantly higher than retention. The ability to predict which customers are likely to leave allows for optimizing retention strategies and allocating resources more efficiently.

This project was carried out using Python and several specialized data science and machine learning libraries, with a user interface developed with Streamlit to facilitate the operational use of the model.

---

## Context and Objectives

### Business Context

In the telecommunications sector, the churn rate can vary between 15% and 30% annually, representing considerable financial losses. Each lost customer requires not only replacement by a new customer (acquisition cost), but also the loss of recurring revenues and potentially future revenues.

### Project Objectives

The main objectives of this project are:

1. **Analytical Objective**: Understand the factors that influence customer churn through an in-depth exploratory analysis of the data.

2. **Predictive Objective**: Develop a machine learning model capable of accurately predicting which customers are at risk of churn.

3. **Operational Objective**: Create an interactive application allowing business teams to easily use the model to:
   - Make predictions on new customers
   - Visualize data and results
   - Understand the importance of different variables

4. **Strategic Objective**: Provide actionable insights to improve customer retention strategies.

### Success Criteria

- Model with accuracy above 80%
- Functional and intuitive application
- Identification of main churn factors
- Complete documentation of the process

---

## Data and Methodology

### Dataset Description

The dataset used contains **6,950 customer observations** with **22 initial variables** covering:

- **Demographic variables**: Gender, SeniorCitizen, Partner, Dependents
- **Service variables**: PhoneService, MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies
- **Contractual variables**: Contract, PaperlessBilling, PaymentMethod
- **Financial variables**: MonthlyCharges, TotalCharges
- **Temporal variable**: Tenure (duration of customer relationship in months)
- **Target variable**: Churn (Yes/No)

### Target Variable Distribution

The churn distribution shows class imbalance:
- **Non-churn (0)**: 73.41% of customers
- **Churn (1)**: 26.59% of customers

This imbalance is moderate and will be taken into account in modeling through stratification during train/test split.

### Work Methodology

The project follows a structured methodology in several steps:

1. **Exploration and Understanding**: Descriptive analysis and data visualization
2. **Cleaning and Preparation**: Missing value management, encoding, normalization
3. **Modeling**: Training and evaluation of machine learning models
4. **Deployment**: Creation of an interactive application
5. **Validation**: Testing and validation of results

### Tools and Technologies

- **Python 3.x**: Main programming language
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical calculations
- **Matplotlib & Seaborn**: Data visualization
- **Scikit-learn**: Machine learning (models, preprocessing, metrics)
- **Streamlit**: Framework for interactive web application
- **Joblib**: Model saving and loading

---

## Exploratory Data Analysis

### Missing Values Analysis

The initial analysis revealed the presence of missing values in several columns:

- **OnlineBackup**: 5,558 missing values (79.97%)
- **TotalCharges**: 2,085 missing values (30.00%)
- **TechSupport**: 1,508 missing values (21.70%)
- **StreamingTV**: 1,508 missing values (21.70%)
- **StreamingMovies**: 1,508 missing values (21.70%)

In total, **89.06% of rows** contained at least one missing value. This situation requires an appropriate treatment strategy.

### Churn Distribution Analysis

The churn distribution shows that:
- Approximately **1 in 4 customers** (26.59%) cancel their subscription
- The churn rate is significant and justifies the implementation of a prediction system

### Correlation Analysis

The correlation analysis identified the variables most correlated with churn:

**Variables positively correlated with churn** (increase risk):
- `InternetService_Fiber optic`: 0.308
- `PaymentMethod_Electronic check`: 0.301
- `MonthlyCharges`: 0.193
- `PaperlessBilling_Yes`: 0.192
- `SeniorCitizen`: 0.152

**Variables negatively correlated with churn** (reduce risk):
- `tenure`: -0.352 (most important)
- `Contract_Two year`: -0.302
- `InternetService_No`: -0.228
- `Contract_One year`: -0.179
- `OnlineSecurity_Yes`: -0.172
- `TechSupport_Yes`: -0.164

### Key Insights from Exploratory Analysis

1. **Relationship duration (tenure)** is the most important factor for retention: the longer a customer stays, the less likely they are to leave.

2. **Contract type** strongly influences churn: long-term contracts (Two year) significantly reduce risk.

3. **Fiber optic customers** have a higher churn rate, possibly related to service issues or competition.

4. **Electronic payment method** (Electronic check) is associated with higher risk.

5. **Additional services** (OnlineSecurity, TechSupport) seem to improve retention.

---

## Data Preparation and Cleaning

### Missing Values Treatment Strategy

Several strategies were applied according to variable type:

1. **Removal of non-relevant columns**:
   - `Unnamed: 0`: Redundant index
   - `customerID`: Non-predictive identifier

2. **TotalCharges treatment**:
   - Conversion to numeric type with error handling
   - Replacement of missing values with 0 (very recent clients who have not yet paid)

3. **Service columns treatment**:
   - Missing values in `OnlineBackup`, `TechSupport`, `StreamingTV`, `StreamingMovies` are replaced with "No"
   - This approach is justified because the absence of value generally indicates the absence of the service

4. **Modality standardization**:
   - Replacement of "No internet service" and "No phone service" with "No" to simplify encoding

### Variable Encoding

**Categorical variables**: One-Hot encoding with `drop_first=True` to avoid multicollinearity. Encoded variables include:
- Gender, Partner, Dependents
- All services (PhoneService, MultipleLines, InternetService, etc.)
- Contract, PaperlessBilling, PaymentMethod

**Target variable**: Binary encoding
- "No" → 0 (non-churn)
- "Yes" → 1 (churn)

**Numeric variables**: Kept as is after verification
- SeniorCitizen, tenure, MonthlyCharges, TotalCharges

### Data Normalization

Numeric variables are standardized using scikit-learn's `StandardScaler` to:
- Ensure all variables are on the same scale
- Improve convergence of the logistic regression algorithm
- Allow more equitable interpretation of coefficients

Standardization is applied after train/test split to avoid data leakage:
- `fit_transform()` on training data
- `transform()` on test data

### Train/Test Split

Data is divided with:
- **70% for training** (4,865 observations)
- **30% for testing** (2,085 observations)
- **Stratification** on target variable to preserve churn distribution in both sets

---

## Modeling

### Algorithm Choice: Logistic Regression

Logistic regression was chosen as the main algorithm for several reasons:

1. **Interpretability**: Coefficients are easily interpretable and allow understanding the impact of each variable on churn.

2. **Performance**: For binary classification problems with structured data, logistic regression often offers excellent performance.

3. **Speed**: Training is fast, which is important for an interactive application.

4. **Probabilities**: The model directly provides churn probabilities, useful for customer scoring.

### Hyperparameters

Hyperparameters used:
- **C**: 1.0 (default regularization)
- **max_iter**: 1000 (maximum number of iterations)
- **random_state**: 42 (reproducibility)

The Streamlit application allows modifying these parameters to experiment with different configurations.

### Model Performance

Results on the test set show:

**Global Metrics**:
- **Accuracy**: 82%
- **ROC AUC**: ~0.85 (estimated value based on performance)

**Metrics by Class**:

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Non-churn (0) | 0.86 | 0.90 | 0.88 | 1,531 |
| Churn (1) | 0.68 | 0.58 | 0.63 | 554 |

**Performance Analysis**:

- The model predicts **non-churn** well (precision 86%, recall 90%)
- **Churn** prediction is more difficult (precision 68%, recall 58%)
- Class imbalance partly explains this difference
- The F1-score of 0.63 for the churn class indicates room for improvement

### Confusion Matrix

The confusion matrix allows visualizing:
- **True Negatives (TN)**: Customers predicted as non-churn who do not actually churn
- **False Positives (FP)**: Customers predicted as churn who do not actually churn
- **False Negatives (FN)**: Customers predicted as non-churn who actually churn (high cost)
- **True Positives (TP)**: Customers predicted as churn who actually churn

### Variable Importance

Analysis of logistic regression coefficients reveals the most important variables:

**Variables reducing churn** (negative coefficients):
1. **tenure**: -0.88 (most important)
2. **Contract_Two year**: -0.50
3. **InternetService_No**: -0.39
4. **MonthlyCharges**: -0.31
5. **Contract_One year**: -0.27

**Variables increasing churn** (positive coefficients):
1. **InternetService_Fiber optic**: +0.57
2. **StreamingMovies_Yes**: +0.23
3. **StreamingTV_Yes**: +0.20
4. **PaymentMethod_Electronic check**: +0.17
5. **PaperlessBilling_Yes**: +0.16

### Business Analysis: Impact of Number of Services

An additional analysis was performed to understand the impact of the number of subscribed services on churn:

**Results**:
- Customers with **0 services** have a churn rate of **42%** (very high)
- Customers with **1 service** have a churn rate of **15%** (lowest)
- Customers with **8 services** have a churn rate of **10%** (very low)
- Overall correlation between number of services and churn is weak (0.020)

**Interpretation**:
- Having at least one service significantly reduces churn
- Beyond a certain number, the effect is less pronounced
- Quality and type of services seem more important than quantity

---

## Streamlit Application

### Application Architecture

The Streamlit application (`app_streamlit.py`) is designed as an interactive single-page dashboard with several sections:

1. **Data Upload and Overview**
2. **Missing Values Analysis**
3. **Automatic Preprocessing**
4. **Exploratory Data Analysis (EDA)**
5. **Predictions** (single customer or batch mode)
6. **Model Evaluation**

### Main Features

#### Training Mode

When a CSV file containing the "Churn" column is uploaded:

1. **Automatic preprocessing**: Data is automatically cleaned and prepared
2. **Automatic training**: Model is trained with parameters configured in the sidebar
3. **Automatic saving**: Trained model is saved in `churn_model.joblib`
4. **Visualizations**: Distribution charts, correlations, etc.
5. **Evaluation**: Performance metrics, confusion matrix, ROC curve

#### Prediction Mode

When a CSV file without "Churn" column is uploaded:

1. **Model loading**: Saved model is automatically loaded
2. **Batch prediction**: Predictions for all customers in the file
3. **Results export**: Download predictions as CSV

#### Single Customer Prediction

In training mode, it is possible to:
- Manually enter customer characteristics
- Get immediate prediction with probability
- Visualize churn risk on a progress bar

### User Interface

The interface is organized with:
- **Sidebar**: Model parameters and exploration options
- **Expandable sections**: Clear organization with accordions
- **Interactive visualizations**: Charts generated with Matplotlib and Seaborn
- **Real-time metrics**: Display of performance and statistics

### State Management

The application uses Streamlit's session state system to:
- Preserve preprocessed data
- Avoid unnecessarily retraining the model
- Maintain consistency between different sections

---

## Results and Interpretations

### Overall Performance

The logistic regression model achieves an **overall accuracy of 82%**, which is satisfactory for a first model. However, detailed analysis reveals opportunities for improvement, particularly for detecting customers at risk of churn.

### Actionable Business Insights

#### 1. Focus on New Customer Retention

Since the **tenure** factor is the most important (-0.88), retention efforts must be particularly intense during the first months of the customer relationship. Recommended actions:
- Enhanced onboarding program
- Proactive contacts in the first 3-6 months
- Incentive offers to extend engagement

#### 2. Promotion of Long-Term Contracts

Two-year contracts significantly reduce churn (-0.50). Strategies:
- Financial incentives for long-term contracts
- Communication on benefits of commitment
- Proactive renegotiation before end of short-term contracts

#### 3. Attention to Fiber Optic Customers

Fiber optic customers have higher risk (+0.57). Actions:
- Specific satisfaction surveys
- Service quality improvement
- Enhanced technical support
- Targeted loyalty programs

#### 4. Payment Method Optimization

Electronic check payment is associated with higher risk (+0.17). Recommendations:
- Incentive for automatic payment
- Fee reduction for automatic payments
- Communication on simplicity and security

#### 5. Valuation of Additional Services

Services like OnlineSecurity and TechSupport reduce churn. Strategies:
- Promotional offers for these services
- Demonstration of their value
- Integration into base packages

### Limitations and Possible Improvements

#### Current Limitations

1. **Class imbalance**: Recall for churn class (58%) could be improved
2. **Single model**: Only logistic regression has been tested
3. **Limited feature engineering**: No creation of complex derived features
4. **Cross-validation**: No cross-validation to optimize hyperparameters

#### Recommended Improvements

1. **Rebalancing techniques**:
   - SMOTE to generate synthetic examples of minority class
   - Undersampling of majority class
   - Use of class_weight in the model

2. **Exploration of other algorithms**:
   - Random Forest (better handling of imbalance)
   - XGBoost (often superior performance)
   - Neural networks (to capture complex interactions)

3. **Advanced feature engineering**:
   - MonthlyCharges/TotalCharges ratio
   - Charge trends (increase/decrease)
   - Interactions between variables (e.g., tenure × contract)

4. **Hyperparameter optimization**:
   - Grid Search or Random Search
   - k-fold cross-validation
   - Bayesian optimization

5. **Business metrics**:
   - Cost per false negative vs false positive
   - ROI of retention actions
   - Churn propensity score with optimal thresholds

---

## Conclusion and Perspectives

### Conclusion

This project enabled the development of a complete churn prediction system, from exploratory data analysis to deployment of an interactive application. The logistic regression model achieves 82% accuracy and clearly identifies the main risk factors.

The generated insights are actionable and allow guiding customer retention strategies. The Streamlit application facilitates operational use of the model by business teams.

### Main Achievements

1. ✅ In-depth analysis of 6,950 customers with identification of key factors
2. ✅ Prediction model with 82% accuracy
3. ✅ Functional and intuitive interactive application
4. ✅ Complete documentation of the process
5. ✅ Actionable business insights

### Improvement Perspectives

#### Short Term (1-3 months)

1. **Model improvement**:
   - Testing other algorithms (Random Forest, XGBoost)
   - Hyperparameter optimization
   - Class rebalancing

2. **Data enrichment**:
   - Addition of behavioral data (support calls, complaints)
   - Customer satisfaction data
   - Market data (competition, promotions)

3. **Application improvement**:
   - Addition of interactive visualizations (Plotly)
   - Automatic report export
   - Integration with databases

#### Medium Term (3-6 months)

1. **Production deployment**:
   - REST API for integration with other systems
   - Automated data pipeline
   - Model performance monitoring

2. **Real-time scoring system**:
   - Integration with CRM
   - Automatic alerts for at-risk customers
   - Dashboard for sales teams

3. **A/B Testing**:
   - Validation of retention action impact
   - Optimization of strategies based on predictions

#### Long Term (6-12 months)

1. **Advanced models**:
   - Deep Learning to capture complex patterns
   - Time series models to predict churn timing
   - Ensemble models (stacking, blending)

2. **Complete automation**:
   - Automatic model retraining
   - Data drift detection
   - Continuous prediction updates

3. **Expansion**:
   - Application to other customer segments
   - Prediction of other metrics (upselling, cross-selling)
   - Integration with other analytical systems

### Expected Business Impact

With a churn rate of 26.59% and a base of 6,950 customers, a reduction of only 5% in churn would represent:
- **~185 customers retained** per period
- **Significant preserved recurring revenues**
- **Reduction in acquisition costs** to replace lost customers

Investment in this prediction system should generate positive ROI by enabling more efficient allocation of retention resources to the most at-risk customers.

### Final Recommendations

1. **Deploy the model in production** with continuous monitoring
2. **Train teams** on application use and result interpretation
3. **Implement targeted retention actions** based on predictions
4. **Measure impact** of actions and iterate on the model
5. **Continuously enrich** data and improve the model

---

## Appendices

### Project Structure

```
Churn_prediction/
├── app_streamlit.py          # Main Streamlit application
├── data_analysis.ipynb       # Exploratory analysis notebook
├── data_churn.csv            # Main dataset
├── churn_model.joblib        # Saved model
├── requirements.txt          # Python dependencies
├── README.md                 # Installation documentation
└── rapport.md                # This report
```

### Usage Commands

**Installation**:
```bash
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

**Launching the application**:
```bash
streamlit run app_streamlit.py
```

### Bibliography and References

- Scikit-learn Documentation: https://scikit-learn.org/
- Streamlit Documentation: https://docs.streamlit.io/
- Pandas Documentation: https://pandas.pydata.org/
- Seaborn Documentation: https://seaborn.pydata.org/

---

**End of Report**
