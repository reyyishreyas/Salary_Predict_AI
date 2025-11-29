import os
import time
import json
import warnings
import pandas as pd
import numpy as np
import joblib

from xgboost import XGBRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, StackingRegressor, VotingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

warnings.filterwarnings('ignore')
os.makedirs('results', exist_ok=True)

print("Starting Ensemble Model Training for Salary Prediction")

# Load datasets
start_total = time.time()
X_train = pd.read_csv('processed/X_train.csv')
X_test = pd.read_csv('processed/X_test.csv')
y_train = pd.read_csv('processed/y_train.csv').squeeze()
y_test = pd.read_csv('processed/y_test.csv').squeeze()

print(f"Training set shape: {X_train.shape}")
print(f"Test set shape: {X_test.shape}")

# Train simple voting ensemble as a baseline
voting_ensemble = VotingRegressor(
    estimators=[
        ('xgb', XGBRegressor(n_estimators=500, max_depth=8, learning_rate=0.1, random_state=42, n_jobs=-1)),
        ('gb', GradientBoostingRegressor(n_estimators=500, max_depth=8, learning_rate=0.1, random_state=42)),
        ('rf', RandomForestRegressor(n_estimators=500, max_depth=18, random_state=42, n_jobs=-1))
    ],
    weights=[3, 2, 1]
)

start_voting = time.time()
voting_ensemble.fit(X_train, y_train)
voting_time = time.time() - start_voting

voting_pred = voting_ensemble.predict(X_test)
voting_r2 = r2_score(y_test, voting_pred)

print(f"Voting Ensemble R²: {voting_r2:.6f}")
print(f"Time taken: {voting_time/60:.2f} minutes")

# Train advanced stacking ensemble
base_models = [
    ('xgb1', XGBRegressor(n_estimators=400, max_depth=7, learning_rate=0.08, random_state=42, n_jobs=-1)),
    ('xgb2', XGBRegressor(n_estimators=400, max_depth=9, learning_rate=0.12, random_state=43, n_jobs=-1)),
    ('gb1', GradientBoostingRegressor(n_estimators=400, max_depth=8, learning_rate=0.08, random_state=42)),
    ('gb2', GradientBoostingRegressor(n_estimators=400, max_depth=10, learning_rate=0.12, random_state=43)),
    ('rf', RandomForestRegressor(n_estimators=600, max_depth=20, random_state=42, n_jobs=-1))
]

meta_learner = Ridge(alpha=1.0, random_state=42)

stacking_ensemble = StackingRegressor(
    estimators=base_models,
    final_estimator=meta_learner,
    cv=5
)

start_stacking = time.time()
stacking_ensemble.fit(X_train, y_train)
stacking_time = time.time() - start_stacking

stacking_pred = stacking_ensemble.predict(X_test)
stacking_r2 = r2_score(y_test, stacking_pred)

print(f"Stacking Ensemble R²: {stacking_r2:.6f}")
print(f"Time taken: {stacking_time/60:.2f} minutes")

# Select best ensemble
if stacking_r2 > voting_r2:
    best_ensemble = stacking_ensemble
    best_type = "Stacking"
    best_r2 = stacking_r2
    y_pred = stacking_pred
    total_time_taken = stacking_time
else:
    best_ensemble = voting_ensemble
    best_type = "Voting"
    best_r2 = voting_r2
    y_pred = voting_pred
    total_time_taken = voting_time

# Evaluate predictions
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

print(f"Ensemble Type: {best_type}")
print(f"R² Score: {best_r2:.6f}")
print(f"MAE: ₹{mae:,.0f}")
print(f"RMSE: ₹{rmse:,.0f}")
print(f"MAPE: {mape:.2f}%")

# Feature importance
if best_type == "Stacking":
    xgb_model = stacking_ensemble.estimators_[0]
    gb_model = stacking_ensemble.estimators_[2]
    rf_model = stacking_ensemble.estimators_[4]
else:
    xgb_model = voting_ensemble.estimators_[0]
    gb_model = voting_ensemble.estimators_[1]
    rf_model = voting_ensemble.estimators_[2]

def normalize(arr):
    return arr / arr.sum() if arr.sum() != 0 else arr

avg_importance = (normalize(xgb_model.feature_importances_) +
                  normalize(gb_model.feature_importances_) +
                  normalize(rf_model.feature_importances_)) / 3

feature_importance = pd.DataFrame({
    'Feature': X_train.columns,
    'XGBoost': xgb_model.feature_importances_,
    'GradientBoosting': gb_model.feature_importances_,
    'RandomForest': rf_model.feature_importances_,
    'Average': avg_importance
}).sort_values('Average', ascending=False)

feature_importance.to_csv('results/feature_importance.csv', index=False)

# Predictions analysis
predictions_df = pd.DataFrame({
    'Actual': y_test,
    'Predicted': y_pred,
    'Difference': y_test - y_pred,
    'Abs_Error': np.abs(y_test - y_pred),
    'Pct_Error': np.abs((y_test - y_pred) / y_test) * 100
})

within_1pct = (predictions_df['Pct_Error'] <= 1).sum()
within_5pct = (predictions_df['Pct_Error'] <= 5).sum()
within_10pct = (predictions_df['Pct_Error'] <= 10).sum()

predictions_df.to_csv('results/test_predictions.csv', index=False)

# Save model and metadata
joblib.dump(best_ensemble, 'models/model.pkl')

metadata = {
    'ensemble_type': best_type,
    'r2_score': float(best_r2),
    'mae': float(mae),
    'rmse': float(rmse),
    'mape': float(mape),
    'voting_r2': float(voting_r2),
    'stacking_r2': float(stacking_r2),
    'total_training_time_minutes': float(total_time_taken / 60),
    'training_samples': len(X_train),
    'test_samples': len(X_test),
    'features': X_train.shape[1],
    'predictions_within_1pct': int(within_1pct),
    'predictions_within_5pct': int(within_5pct),
    'predictions_within_10pct': int(within_10pct)
}

with open('results/model_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=4)

total_time = time.time() - start_total

# Final summary
print("Training Complete")
print(f"Ensemble: {best_type}")
print(f"R² Score: {best_r2*100:.2f}%")
print(f"MAE: ₹{mae:,.0f}, RMSE: ₹{rmse:,.0f}, MAPE: {mape:.2f}%")
print(f"Predictions within 5% error: {within_5pct/len(predictions_df)*100:.1f}%")
print(f"Total training time: {total_time/60:.2f} minutes")