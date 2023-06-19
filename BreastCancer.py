
'''
Supervised machine learning to analyze breast cancer diagnoses.
Methods used:
1. Logistic Regression
2. Random Forest
3. Naive Bayes
'''


import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns

# Load the Breast Cancer dataset
breast_cancer = load_breast_cancer()
data = pd.DataFrame(data=breast_cancer.data, columns=breast_cancer.feature_names)
data['Target'] = breast_cancer.target

# Separate the features and target variable
X = data.drop("Target", axis=1)
y = data["Target"]

# Create logistic regression, random forest, and naive Bayes models
logreg_model = LogisticRegression()
rf_model = RandomForestClassifier()
nb_model = GaussianNB()

# Perform cross-validation and obtain predictions for each model
logreg_predictions = cross_val_predict(logreg_model, X, y, cv=10)
rf_predictions = cross_val_predict(rf_model, X, y, cv=10)
nb_predictions = cross_val_predict(nb_model, X, y, cv=10)

# Calculate accuracy scores for each model
logreg_accuracy = accuracy_score(y, logreg_predictions)
rf_accuracy = accuracy_score(y, rf_predictions)
nb_accuracy = accuracy_score(y, nb_predictions)

# Create confusion matrices for each model
logreg_cm = confusion_matrix(y, logreg_predictions)
rf_cm = confusion_matrix(y, rf_predictions)
nb_cm = confusion_matrix(y, nb_predictions)

# Create a DataFrame to display the results
results_df = pd.DataFrame(
    {'Algorithm': ['Logistic Regression', 'Random Forest', 'Naive Bayes'],
     'Accuracy': [logreg_accuracy, rf_accuracy, nb_accuracy],
     'Confusion Matrix': [logreg_cm, rf_cm, nb_cm]}
)

# Display the results DataFrame
print(results_df)