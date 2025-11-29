import pandas as pd
import numpy as np
import time
import json
import os
import warnings

from sklearn.linear_model import ElasticNet, Ridge
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import joblib

warnings.filterwarnings("ignore")

# Prepare results folder
os.makedirs("results/linear_regression", exist_ok=True)

# Load Dataset
print("Loading training and test datasets...")

start_time = time.time()

X_train = pd.read_csv("processed/X_train.csv")
X_test = pd.read_csv("processed/X_test.csv")
y_train = pd.read_csv("processed/y_train.csv").squeeze()
y_test = pd.read_csv("processed/y_test.csv").squeeze()

print(f"Training set:  {X_train.shape}")
print(f"Test set:      {X_test.shape}")

# Baseline Model for Comparison (Ridge Regression)
print("\nTraining baseline Ridge model...")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

baseline = Ridge(alpha=1.0)
baseline.fit(X_train_scaled, y_train)

baseline_pred = baseline.predict(X_test_scaled)
baseline_r2 = r2_score(y_test, baseline_pred)

print(f"Baseline R²: {baseline_r2:.4f}")

# Build Pipeline (Polynomial Features + Scaling + ElasticNet)
print("\nSetting up ElasticNet pipeline with polynomial features...")

model = Pipeline([
    ("poly", PolynomialFeatures(include_bias=False)),
    ("scaler", StandardScaler()),
    ("elasticnet", ElasticNet(max_iter=2000, random_state=42))
])


# Hyperparameter Tuning
print("Starting GridSearch hyperparameter tuning...")

param_grid = {
    "poly__degree": [2, 3],
    "elasticnet__alpha": [0.1, 1.0, 10.0, 50.0],
    "elasticnet__l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9]
}

grid = GridSearchCV(
    model,
    param_grid,
    scoring="r2",
    cv=5,
    n_jobs=-1,
    verbose=1
)

tuning_start = time.time()
grid.fit(X_train, y_train)
tuning_duration = time.time() - tuning_start

print(f"Tuning completed in {tuning_duration/60:.2f} minutes")
print("Best parameters:", grid.best_params_)
print(f"Best CV R²: {grid.best_score_:.4f}")

# Final Evaluation

print("\nEvaluating tuned model on test data...")

best_model = grid.best_estimator_
pred = best_model.predict(X_test)

r2 = r2_score(y_test, pred)
mae = mean_absolute_error(y_test, pred)
rmse = np.sqrt(mean_squared_error(y_test, pred))
mape = np.mean(np.abs((y_test - pred) / y_test)) * 100

print(f"Model R²:   {r2:.4f}")
print(f"MAE:        {mae:,.0f}")
print(f"RMSE:       {rmse:,.0f}")
print(f"MAPE:       {mape:.2f}%")
print(f"Improvement over baseline: {(r2 - baseline_r2) * 100:.2f}%")

# --------------------------------------------------------------------
# Feature Importance Extraction
# --------------------------------------------------------------------
print("\nExtracting feature contributions...")

poly = best_model.named_steps["poly"]
enet = best_model.named_steps["elasticnet"]

feature_names = poly.get_feature_names_out(X_train.columns)

coeff_df = pd.DataFrame({
    "Feature": feature_names,
    "Coefficient": enet.coef_,
    "Abs_Value": np.abs(enet.coef_)
}).sort_values("Abs_Value", ascending=False)

non_zero = coeff_df[coeff_df["Abs_Value"] > 0]

coeff_df.to_csv("results/linear_regression/feature_coefficients.csv", index=False)

print(f"Polynomial features generated: {len(feature_names)}")
print(f"Features retained (non-zero coeffs): {len(non_zero)}")

#  Save Model and Outputs

print("\nSaving model outputs...")

joblib.dump(best_model, "results/linear_regression/model_pipeline.pkl")

pred_df = pd.DataFrame({
    "Actual": y_test,
    "Predicted": pred,
    "Error": y_test - pred,
    "Absolute_Error": np.abs(y_test - pred),
    "Percentage_Error": np.abs((y_test - pred) / y_test) * 100
})
pred_df.to_csv("results/linear_regression/test_predictions.csv", index=False)

metadata = {
    "model": "ElasticNet + Polynomial Features",
    "baseline_r2": float(baseline_r2),
    "final_r2": float(r2),
    "mae": float(mae),
    "rmse": float(rmse),
    "mape": float(mape),
    "tuning_time_sec": float(tuning_duration),
    "best_params": grid.best_params_,
    "original_features": X_train.shape[1],
    "poly_features": len(feature_names),
    "selected_features": len(non_zero),
}

with open("results/linear_regression/model_metadata.json", "w") as f:
    json.dump(metadata, f, indent=4)

total = time.time() - start_time
print(f"\nTotal runtime: {total/60:.2f} minutes")
print("Training complete. All files saved in results/linear_regression/")
