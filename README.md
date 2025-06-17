# 📞 Telco Customer Churn Prediction

This project predicts whether a customer will churn (leave the service) based on their service usage, contract, and demographic data. It uses various classification algorithms and is deployed as a user-friendly Streamlit app.

🔗 **Live App**: [Telco Churn Predictor](https://telco-customer-churn-prediction-hwaqngpssf95kxlhh7ena2.streamlit.app/)

---

## 📁 Project Structure


---

## 📊 Dataset Overview

- **Source**: IBM Sample Dataset (via Kaggle or IBM repository)
- **Total Records**: ~7,000 customers
- **Features** include:
  - Customer Demographics (gender, senior citizen, etc.)
  - Service Subscriptions (Internet, Phone, etc.)
  - Contract Type, Payment Method
  - Monthly Charges, Total Charges
- **Target**: `Churn` (Yes/No)

---

## 📈 Data Processing & Modeling

### 🧹 Preprocessing
- Categorical encoding (LabelEncoder / One-Hot)
- Null value treatment and data cleaning
- Feature scaling using `StandardScaler`

### 📊 Exploratory Data Analysis (EDA)
- Churn distribution visualization
- Correlation heatmaps
- Contract & service impact analysis

### 🤖 Models Trained
- Logistic Regression
- Random Forest Classifier
- Decision Tree Classifier
- Support Vector Machine (SVM)
- Gradient Boosting

🏆 **Best Model** (based on accuracy/F1): Random Forest or Gradient Boosting (update if needed)

---

## 🚀 Try the Web App

Use the app live here:  
👉 [https://telco-customer-churn-prediction-hwaqngpssf95kxlhh7ena2.streamlit.app/](https://telco-customer-churn-prediction-hwaqngpssf95kxlhh7ena2.streamlit.app/)


