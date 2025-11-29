import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

# Ensure results folder exists
os.makedirs('results', exist_ok=True)

# Load model and test data
model = joblib.load('models/model.pkl')
X_test = pd.read_csv('processed/X_test.csv')
y_test = pd.read_csv('processed/y_test.csv').squeeze()

# Make predictions
y_pred = model.predict(X_test)

# Prepare a DataFrame for analysis
results = pd.DataFrame({
    'Actual': y_test,
    'Predicted': y_pred
})
results['Error'] = results['Actual'] - results['Predicted']
results['Abs_Error'] = results['Error'].abs()
results['Perc_Error'] = results['Abs_Error'] / results['Actual'] * 100

# 1. Actual vs Predicted
plt.figure(figsize=(8,6))
sns.scatterplot(x='Actual', y='Predicted', data=results, alpha=0.6, color='#1f77b4')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Actual Salary')
plt.ylabel('Predicted Salary')
plt.title('Actual vs Predicted')
plt.tight_layout()
plt.savefig('results/actual_vs_predicted.png')
plt.close()

# 2. Error distribution
plt.figure(figsize=(8,6))
sns.histplot(results['Error'], bins=30, kde=True, color='#ff7f0e')
plt.xlabel('Prediction Error')
plt.ylabel('Frequency')
plt.title('Error Distribution')
plt.tight_layout()
plt.savefig('results/error_distribution.png')
plt.close()

# 3. Absolute error vs Actual
plt.figure(figsize=(8,6))
sns.scatterplot(x='Actual', y='Abs_Error', data=results, alpha=0.6, color='#2ca02c')
plt.xlabel('Actual Salary')
plt.ylabel('Absolute Error')
plt.title('Absolute Error vs Actual')
plt.tight_layout()
plt.savefig('results/abs_error_vs_actual.png')
plt.close()

# 4. Percentage error distribution
plt.figure(figsize=(8,6))
sns.histplot(results['Perc_Error'], bins=30, kde=True, color='#d62728')
plt.xlabel('Percentage Error (%)')
plt.ylabel('Frequency')
plt.title('Percentage Error')
plt.tight_layout()
plt.savefig('results/percentage_error.png')
plt.close()

# 5. Predictions over sample index
plt.figure(figsize=(10,5))
plt.plot(results['Actual'].values[:100], label='Actual', marker='o')
plt.plot(results['Predicted'].values[:100], label='Predicted', marker='x')
plt.xlabel('Sample Index')
plt.ylabel('Salary')
plt.title('Predictions vs Sample Index (first 100)')
plt.legend()
plt.tight_layout()
plt.savefig('results/predictions_vs_index.png')
plt.close()

# 6. Feature importance
if hasattr(model, 'feature_importances_'):
    importance = pd.DataFrame({
        'Feature': X_test.columns,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False).head(15)
    
    plt.figure(figsize=(8,6))
    sns.barplot(x='Importance', y='Feature', data=importance, palette='viridis')
    plt.title('Top 15 Feature Importances')
    plt.tight_layout()
    plt.savefig('results/feature_importance.png')
    plt.close()

print("All evaluation plots have been saved in the 'results/' folder.")
