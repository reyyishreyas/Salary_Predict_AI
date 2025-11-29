import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import json
import warnings

warnings.filterwarnings("ignore")

print("\nTraining XGBoost Salary Prediction Model\n")

# Load train/test datasets
print("Loading datasets...")

X_train = pd.read_csv("processed/X_train.csv")
X_test = pd.read_csv("processed/X_test.csv")
y_train = pd.read_csv("processed/y_train.csv").squeeze()
y_test = pd.read_csv("processed/y_test.csv").squeeze()

print(f"Training data shape : {X_train.shape}")
print(f"Test data shape     : {X_test.shape}")
print(f"Total features      : {X_train.shape[1]}\n")

# Baseline model
print("Training initial baseline model...")

baseline_model = XGBRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

baseline_model.fit(X_train, y_train)
baseline_preds = baseline_model.predict(X_test)

baseline_r2 = r2_score(y_test, baseline_preds)
baseline_mae = mean_absolute_error(y_test, baseline_preds)
baseline_rmse = np.sqrt(mean_squared_error(y_test, baseline_preds))

print("\nBaseline Performance:")
print(f"R² Score : {baseline_r2:.4f}")
print(f"MAE      : ₹{baseline_mae:,.0f}")
print(f"RMSE     : ₹{baseline_rmse:,.0f}\n")

# Hyperparameter tuning
print("Tuning hyperparameters (this may take a while)...")

param_grid = {
    "n_estimators": [300, 500, 700],
    "max_depth": [6, 8, 10],
    "learning_rate": [0.01, 0.05, 0.1],
    "min_child_weight": [1, 3, 5],
    "subsample": [0.8, 0.9, 1.0],
    "colsample_bytree": [0.8, 0.9, 1.0],
    "gamma": [0, 0.1, 0.2]
}

grid = GridSearchCV(
    estimator=XGBRegressor(random_state=42, n_jobs=-1),
    param_grid=param_grid,
    cv=5,
    scoring="r2",
    n_jobs=-1,
    verbose=1
)

grid.fit(X_train, y_train)

print("\nBest Parameters:")
for k, v in grid.best_params_.items():
    print(f"• {k}: {v}")

print(f"\nBest CV R² Score: {grid.best_score_:.4f}\n")

# Tuned model evaluation
print("Evaluating tuned model...")

tuned_model = grid.best_estimator_
tuned_preds = tuned_model.predict(X_test)

tuned_r2 = r2_score(y_test, tuned_preds)
tuned_mae = mean_absolute_error(y_test, tuned_preds)
tuned_rmse = np.sqrt(mean_squared_error(y_test, tuned_preds))
tuned_mape = np.mean(np.abs((y_test - tuned_preds) / y_test)) * 100

print("Tuned Model Performance:")
print(f"R² Score : {tuned_r2:.4f}")
print(f"MAE      : ₹{tuned_mae:,.0f}")
print(f"RMSE     : ₹{tuned_rmse:,.0f}")
print(f"MAPE     : {tuned_mape:.2f}%\n")

# Cross-validation
print("Running 5-fold cross validation...")

cv_scores = cross_val_score(
    tuned_model, X_train, y_train,
    scoring="r2",
    cv=5,
    n_jobs=-1
)

print(f"CV Scores : {cv_scores}")
print(f"Mean R²   : {cv_scores.mean():.4f}")
print(f"Std Dev   : {cv_scores.std():.4f}\n")

# Feature importance
print("Extracting feature importance...")

importance_df = pd.DataFrame({
    "Feature": X_train.columns,
    "Importance": tuned_model.feature_importances_
}).sort_values("Importance", ascending=False)

print("\nTop 15 Important Features:")
print(importance_df.head(15).to_string(index=False))

importance_df.to_csv("results/feature_importance.csv", index=False)

# Saving outputs
print("\nSaving model and output files...")

joblib.dump(tuned_model, "results/xgboost_salary_model.pkl")

metadata = {
    "model_type": "XGBRegressor",
    "training_samples": len(X_train),
    "test_samples": len(X_test),
    "features": X_train.shape[1],
    "baseline_r2": float(baseline_r2),
    "tuned_r2": float(tuned_r2),
    "mae": float(tuned_mae),
    "rmse": float(tuned_rmse),
    "mape": float(tuned_mape),
    "cv_mean_r2": float(cv_scores.mean()),
    "cv_std_r2": float(cv_scores.std()),
    "best_params": grid.best_params_
}

with open("results/model_metadata.json", "w") as f:
    json.dump(metadata, f, indent=4)

pred_df = pd.DataFrame({
    "Actual_Salary": y_test,
    "Predicted_Salary": tuned_preds,
    "Difference": y_test - tuned_preds,
    "Absolute_Error": np.abs(y_test - tuned_preds),
    "Percentage_Error": np.abs((y_test - tuned_preds) / y_test) * 100
})

pred_df.to_csv("results/test_predictions.csv", index=False)

# Final summary
print("\nTraining Complete.\n")

print("Summary:")
print(f"Baseline R² : {baseline_r2*100:.2f}%")
print(f"Tuned R²    : {tuned_r2*100:.2f}%")
print(f"Improvement : {(tuned_r2 - baseline_r2)*100:.2f}%\n")

print("Files generated:")
print(" - results/xgboost_salary_model.pkl")
print(" - results/model_metadata.json")
print(" - results/feature_importance.csv")
print(" - results/test_predictions.csv\n")

print("Model is ready to be used for salary predictions.\n")
