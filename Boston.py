
'''
Supervised machine learning to analyze boston dataset.
Methods used:
1. Linear Regression
2. Decision Tree
3. Random Forest
'''

# Import libraries
import numpy as np
import pandas as pd
from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load the Boston housing dataset
boston = load_boston()
data = pd.DataFrame(data=boston.data, columns=boston.feature_names)
data['TargetVariable'] = boston.target

# Separate the features and target variable
X = data.drop("TargetVariable", axis=1)
y = data["TargetVariable"]

# Create linear regression, decision tree, and random forest models
linear_model = LinearRegression()
tree_model = DecisionTreeRegressor()
forest_model = RandomForestRegressor()

# Perform cross-validation and obtain predictions for each model
linear_predictions = cross_val_predict(linear_model, X, y, cv=10)
tree_predictions = cross_val_predict(tree_model, X, y, cv=10)
forest_predictions = cross_val_predict(forest_model, X, y, cv=10)

# Calculate evaluation metrics for each model
linear_mse = mean_squared_error(y, linear_predictions)
linear_rmse = np.sqrt(linear_mse)
linear_r_squared = r2_score(y, linear_predictions)

tree_mse = mean_squared_error(y, tree_predictions)
tree_rmse = np.sqrt(tree_mse)
tree_r_squared = r2_score(y, tree_predictions)

forest_mse = mean_squared_error(y, forest_predictions)
forest_rmse = np.sqrt(forest_mse)
forest_r_squared = r2_score(y, forest_predictions)

# Print evaluation metrics for each model
print("Linear Regression:")
print("Mean Squared Error (MSE):", linear_mse)
print("Root Mean Squared Error (RMSE):", linear_rmse)
print("R-squared:", linear_r_squared)
print()

print("Decision Tree:")
print("Mean Squared Error (MSE):", tree_mse)
print("Root Mean Squared Error (RMSE):", tree_rmse)
print("R-squared:", tree_r_squared)
print()

print("Random Forest:")
print("Mean Squared Error (MSE):", forest_mse)
print("Root Mean Squared Error (RMSE):", forest_rmse)
print("R-squared:", forest_r_squared)
print()

# Plot the actual vs. predicted values for each model
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.scatter(y, linear_predictions)
plt.plot([y.min(), y.max()], [y.min(), y.max()], color='red', linestyle='--')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Linear Regression')

plt.subplot(1, 3, 2)
plt.scatter(y, tree_predictions)
plt.plot([y.min(), y.max()], [y.min(), y.max()], color='red', linestyle='--')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Decision Tree')

plt.subplot(1, 3, 3)
plt.scatter(y, forest_predictions)
plt.plot([y.min(), y.max()], [y.min(), y.max()], color='red', linestyle='--')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Random Forest')

plt.tight_layout()
plt.show()