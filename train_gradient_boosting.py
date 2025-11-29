"""
Gradient Boosting Regressor – Training Pipeline
This script trains a tuned Gradient Boosting model, evaluates performance,
generates learning curves, extracts feature importance, and saves results
into the results/gradient_boosting directory.
"""

import os
import time
import json
import warnings
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error
)

warnings.filterwarnings("ignore")

OUTPUT_DIR = "results/gradient_boosting"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# 1. Load training and test data
print("Loading data...")

X_train = pd.read_csv("processed/X_train.csv")
X_test = pd.read_csv("processed/X_test.csv")
y_train = pd.read_csv("processed/y_train.csv").squeeze()
y_test = pd.read_csv("processed/y_test.csv").squeeze()

print(f"Training samples: {X_train.shape}")
print(f"Test samples:     {X_test.shape}")
print(f"Total features:    {X_train.shape[1]}")


# 2. Baseline model for quick benchmark
print("\nTraining baseline GradientBoosting model...")

baseline_model = GradientBoostingRegressor(
    n_estimators=100,
    random_state=42
)

baseline_start = time.time()
baseline_model.fit(X_train, y_train)
baseline_time = time.time() - baseline_start

baseline_pred = baseline_model.predict(X_test)

baseline_r2 = r2_score(y_test, baseline_pred)
baseline_mae = mean_absolute_error(y_test, baseline_pred)

print(f"Baseline R²:  {baseline_r2:.4f}")
print(f"Baseline MAE: {baseline_mae:.2f}")
print(f"Training time: {baseline_time:.2f} seconds")


# 3. Hyperparameter tuning
print("\nRunning GridSearchCV (this may take a while)...")

param_grid = {
    "n_estimators": [500, 700, 1000],
    "max_depth": [7, 9, 11],
    "learning_rate": [0.05, 0.1],
    "min_samples_split": [2, 5],
    "subsample": [0.8, 1.0]
}

grid = GridSearchCV(
    GradientBoostingRegressor(random_state=42),
    param_grid=param_grid,
    cv=3,
    scoring="r2",
    n_jobs=-1,
    verbose=1
)

search_start = time.time()
grid.fit(X_train, y_train)
search_time = time.time() - search_start

print("Best parameters:")
for k, v in grid.best_params_.items():
    print(f"  {k}: {v}")

print(f"Best CV score: {grid.best_score_:.4f}")
print(f"Grid search time: {search_time/60:.2f} minutes")


# 4. Evaluate tuned model
print("\nEvaluating tuned model...")

best_model = grid.best_estimator_
y_pred = best_model.predict(X_test)

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

print(f"Test R²:  {r2:.4f}")
print(f"MAE:      {mae:.2f}")
print(f"RMSE:     {rmse:.2f}")
print(f"MAPE:     {mape:.2f}%")

improvement = (r2 - baseline_r2) * 100
print(f"Improvement vs baseline: {improvement:.2f}%")


# 5. Feature importance
print("\nExtracting feature importance...")

importance_df = pd.DataFrame({
    "Feature": X_train.columns,
    "Importance": best_model.feature_importances_
}).sort_values("Importance", ascending=False)

importance_df.to_csv(f"{OUTPUT_DIR}/feature_importance.csv", index=False)


# 6. Learning curve data (varying n_estimators)
print("Generating learning curve data...")

estimator_range = [50, 100, 200, 300, 400, 500, 700, 1000]
train_scores, test_scores = [], []

for n in estimator_range:
    model = GradientBoostingRegressor(
        n_estimators=n,
        max_depth=grid.best_params_["max_depth"],
        learning_rate=grid.best_params_["learning_rate"],
        min_samples_split=grid.best_params_["min_samples_split"],
        subsample=grid.best_params_["subsample"],
        random_state=42
    )

    model.fit(X_train, y_train)

    train_scores.append(r2_score(y_train, model.predict(X_train)))
    test_scores.append(r2_score(y_test, model.predict(X_test)))

curve_df = pd.DataFrame({
    "n_estimators": estimator_range,
    "train_r2": train_scores,
    "test_r2": test_scores
})

curve_df.to_csv(f"{OUTPUT_DIR}/learning_curves.csv", index=False)


# 7. Save model, predictions, and metadata
print("Saving model and outputs...")

joblib.dump(best_model, f"{OUTPUT_DIR}/model.pkl")

pred_df = pd.DataFrame({
    "Actual": y_test,
    "Predicted": y_pred,
    "Error": y_test - y_pred,
    "Absolute_Error": np.abs(y_test - y_pred),
    "Percentage_Error": np.abs((y_test - y_pred) / y_test) * 100
})

pred_df.to_csv(f"{OUTPUT_DIR}/test_predictions.csv", index=False)

metadata = {
    "r2": float(r2),
    "mae": float(mae),
    "rmse": float(rmse),
    "mape": float(mape),
    "baseline_r2": float(baseline_r2),
    "baseline_mae": float(baseline_mae),
    "improvement_percent": float(improvement),
    "training_samples": len(X_train),
    "test_samples": len(X_test),
    "features": X_train.shape[1],
    "best_params": grid.best_params_,
    "training_time_minutes": search_time / 60
}

with open(f"{OUTPUT_DIR}/model_metadata.json", "w") as f:
    json.dump(metadata, f, indent=4)

print("Training complete. Files saved in results/gradient_boosting/")
