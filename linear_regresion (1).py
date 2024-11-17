# Importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# 1. Load the Boston Housing Dataset
# Load the dataset from CSV file
df = pd.read_csv('HousingData.csv')  # Ensure that this file is in the same directory as the script or provide the full path

# 2. Data Exploration: Display the first few rows and basic statistics
print("First few rows of the dataset:\n", df.head())
print("\nBasic statistics of the dataset:\n", df.describe())

# Visualizing the correlation matrix
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt='.2f')
plt.title('Correlation Matrix')
plt.show()

# 3. Handle Missing Values
# Checking for missing values
print("\nMissing values in each column:\n", df.isnull().sum())

# Impute missing values with the mean (you can also choose to drop the rows or use other methods)
df.fillna(df.mean(), inplace=True)

# 4. Data Preprocessing
# Split dataset into features (X) and target (y)
X = df.drop('MEDV', axis=1)  # Independent variables (drop the target 'MEDV')
y = df['MEDV']  # Dependent variable (target)

# Split the dataset into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Train the Linear Regression Model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict the target values using the test set
y_pred = model.predict(X_test)

# 6. Model Evaluation
mse = mean_squared_error(y_test, y_pred)  # Mean Squared Error
mae = mean_absolute_error(y_test, y_pred)  # Mean Absolute Error
r2 = r2_score(y_test, y_pred)  # R-squared (Coefficient of Determination)

# Print model performance metrics
print("\nModel Performance Metrics:")
print(f"Mean Squared Error (MSE): {mse}")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"R-squared: {r2}")

# 7. Visualization: Actual vs Predicted
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, color='blue', alpha=0.5)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', lw=2)  # Diagonal line for perfect prediction
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted House Prices')
plt.show()

# 8. Residual Analysis
residuals = y_test - y_pred

# Plot residuals vs predicted values
plt.figure(figsize=(8, 6))
plt.scatter(y_pred, residuals, color='blue', alpha=0.5)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.show()

# 9. Coefficients and Intercept Interpretation
print("\nModel Coefficients and Intercept:")
print("Intercept:", model.intercept_)
print("Coefficients:", model.coef_)

# 10. Save Results to CSV
results_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
results_df.to_csv('boston_housing_results.csv', index=False)
print("\nResults saved to 'boston_housing_results.csv'")
