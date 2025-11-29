import pandas as pd
import numpy as np
import time
import json
import os
import warnings

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import joblib

warnings.filterwarnings("ignore")

# Prepare output folder
os.makedirs("results/random_forest", exist_ok=True)

# Load Data

print("Loading datasets...")

start_time = time.time()

X_train = pd.read_csv("processed/X_train.csv")
X_test = pd.read_csv("processed/X_test.csv")
y_train = pd.read_csv("processed/y_train.csv").squeeze()
y_test = pd.read_csv("processed/y_test.csv").squeeze()

print(f"Training data: {X_train.shape}")
print(f"Test data:     {X_test.shape}")


# Train Model

print("\nTraining Random Forest model...")

model = RandomForestRegressor(
    n_estimators=500,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features="sqrt",
    n_jobs=-1,
    random_state=42,
    verbose=1
)

train_start = time.time()
model.fit(X_train, y_train)
train_duration = time.time() - train_start

print(f"Training completed in {train_duration:.1f} seconds")


# Evaluation
print("\nEvaluating model...")

pred = model.predict(X_test)

r2 = r2_score(y_test, pred)
mae = mean_absolute_error(y_test, pred)
rmse = np.sqrt(mean_squared_error(y_test, pred))
mape = np.mean(np.abs((y_test - pred) / y_test)) * 100

print(f"R² Score: {r2:.4f}")
print(f"MAE:      {mae:,.0f}")
print(f"RMSE:     {rmse:,.0f}")
print(f"MAPE:     {mape:.2f}%")


# Save Outputs

print("\nSaving results...")

# Save model
joblib.dump(model, "results/random_forest/model.pkl")


pred_df = pd.DataFrame({
    "Actual": y_test,
    "Predicted": pred,
    "Error": y_test - pred,
    "Absolute_Error": np.abs(y_test - pred),
    "Percentage_Error": np.abs((y_test - pred) / y_test) * 100
})
pred_df.to_csv("results/random_forest/test_predictions.csv", index=False)

feat_imp = pd.DataFrame({
    "Feature": X_train.columns,
    "Importance": model.feature_importances_
}).sort_values("Importance", ascending=False)
feat_imp.to_csv("results/random_forest/feature_importance.csv", index=False)

metadata = {
    "model": "Random Forest Regressor",
    "r2_score": float(r2),
    "mae": float(mae),
    "rmse": float(rmse),
    "mape": float(mape),
    "training_time_sec": float(train_duration),
    "n_estimators": 500,
    "max_depth": None,
    "train_samples": len(X_train),
    "test_samples": len(X_test)
}

with open("results/random_forest/model_metadata.json", "w") as f:
    json.dump(metadata, f, indent=4)

total_time = time.time() - start_time
print(f"Total runtime: {total_time:.1f} seconds")

print("\nRandom Forest training complete. Outputs saved in results/random_forest/")
