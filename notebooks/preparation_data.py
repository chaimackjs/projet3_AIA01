import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


data = pd.read_csv("../data/WA_Fn-UseC_-Telco-Customer-Churn.csv")

print(data.head())

print(data.info())

print(data["gender"])

data = data.drop("customerID", axis=1)  

# Encode categorical columns
data["gender"] = data["gender"].map({"Male": 0, "Female": 1})
data["Partner"] = data["Partner"].map({"Yes": 1, "No": 0})
data["Dependents"] = data["Dependents"].map({"Yes": 1, "No": 0})
data["PhoneService"] = data["PhoneService"].map({"Yes": 1, "No": 0})
data["MultipleLines"] = data["MultipleLines"].map({"Yes": 1, "No": 0, "No phone service": 2})
data["InternetService"] = data["InternetService"].map({"DSL": 1, "No": 0, "Fiber optic": 2})
data["OnlineSecurity"] = data["OnlineSecurity"].map({"Yes": 1, "No": 0, "No internet service": 2})
data["OnlineBackup"] = data["OnlineBackup"].map({"Yes": 1, "No": 0, "No internet service": 2})
data["DeviceProtection"] = data["DeviceProtection"].map({"Yes": 1, "No": 0, "No internet service": 2})
data["TechSupport"] = data["TechSupport"].map({"Yes": 1, "No": 0, "No internet service": 2})
data["StreamingTV"] = data["StreamingTV"].map({"Yes": 1, "No": 0, "No internet service": 2})
data["StreamingMovies"] = data["StreamingMovies"].map({"Yes": 1, "No": 0, "No internet service": 2})
data["Contract"] = data["Contract"].map({"Month-to-month": 0, "One year": 1, "Two year": 2})
data["PaperlessBilling"] = data["PaperlessBilling"].map({"Yes": 1, "No": 0})
data["PaymentMethod"] = data["PaymentMethod"].map({
    "Electronic check": 0,
    "Mailed check": 1,
    "Bank transfer (automatic)": 2,
    "Credit card (automatic)": 3
})
data["Churn"] = data["Churn"].map({"Yes": 1, "No": 0})

# Convert TotalCharges to numeric, coerce errors to NaN and fill with 0
data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors='coerce').fillna(0)

# Compute correlation matrix for all columns
correlation_mat = data.corr()

print(correlation_mat)

sns.heatmap(correlation_mat, annot = True)
plt.title("Correlation matrix of Breast Cancer data")
plt.xlabel("cell nucleus features")
plt.ylabel("cell nucleus features")
plt.show()