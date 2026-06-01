# ==============================
# Energy Consumption Prediction
# Complete Project in One Cell
# ==============================

# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load Dataset
# Change the file path according to your dataset location
data = pd.read_csv(r"C:\Users\user\Desktop\Capstone_Project\cleaned_energy_consumption_dataset.csv")

# Display Dataset Information
print("First 5 Rows:")
print(data.head())

print("\nDataset Info:")
print(data.info())

print("\nDataset Statistics:")
print(data.describe())

# Handle Categorical Data
encoder = LabelEncoder()
data['DayType'] = encoder.fit_transform(data['DayType'])

# Feature Selection
X = data[['Hour', 'Temperature', 'Humidity', 'WindSpeed', 'DayType']]
y = data['EnergyConsumption']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =====================================
# Linear Regression Model
# =====================================
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

y_pred_lr = lr_model.predict(X_test)

print("\n===== Linear Regression Results =====")
print("MAE:", mean_absolute_error(y_test, y_pred_lr))
print("MSE:", mean_squared_error(y_test, y_pred_lr))
print("R2 Score:", r2_score(y_test, y_pred_lr))

# =====================================
# Random Forest Regressor Model
# =====================================
rf_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_test)

print("\n===== Random Forest Results =====")
print("MAE:", mean_absolute_error(y_test, y_pred_rf))
print("MSE:", mean_squared_error(y_test, y_pred_rf))
print("R2 Score:", r2_score(y_test, y_pred_rf))

# =====================================
# Visualizations
# =====================================

# Energy Consumption Over Time
plt.figure(figsize=(10,5))
plt.plot(data['EnergyConsumption'])
plt.title('Energy Consumption Over Time')
plt.xlabel('Records')
plt.ylabel('Energy Consumption')
plt.show()

# Temperature vs Energy Consumption
plt.figure(figsize=(8,5))
sns.scatterplot(
    x=data['Temperature'],
    y=data['EnergyConsumption']
)
plt.title('Temperature vs Energy Consumption')
plt.show()

# Correlation Heatmap
plt.figure(figsize=(10,6))
sns.heatmap(
    data.corr(numeric_only=True),
    annot=True
)
plt.title('Correlation Heatmap')
plt.show()

# =====================================
# Sample Prediction
# =====================================
sample_input = pd.DataFrame(
    [[14, 30, 65, 12, 1]],
    columns=['Hour', 'Temperature', 'Humidity', 'WindSpeed', 'DayType']
)

prediction = rf_model.predict(sample_input)

print("\nPredicted Energy Consumption:", prediction[0])

import joblib

# Save Random Forest Model
joblib.dump(rf_model, "energy_model.pkl")

# Save Label Encoder
joblib.dump(encoder, "label_encoder.pkl")

print("Model saved successfully!")