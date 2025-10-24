# scripts/evaluate_model.py
import sys
import os
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.preprocessing import preprocess, create_sequences
from src.utils import load_fd002_dataset

# ---------------- Load dataset ---------------- #
print("📥 Loading FD002 dataset...")
df = load_fd002_dataset()

# ---------------- Split into train/test units ---------------- #
unique_units = df["unit_number"].unique()
split_idx = int(0.8 * len(unique_units))
train_units = unique_units[:split_idx]
test_units = unique_units[split_idx:]

df_test = df[df["unit_number"].isin(test_units)].reset_index(drop=True)
print(f"🧩 Test set units: {len(test_units)} ({len(df_test)} rows)")

# ---------------- Preprocess ---------------- #
print("⚙️ Preprocessing test dataset...")
# Only scale features; load scalers saved during training
scaler_x = joblib.load("models/feature_scaler.pkl")
scaler_y = joblib.load("models/rul_scaler.pkl")

feature_cols = [c for c in df.columns if c not in ["unit_number", "time_cycles", "RUL"]]
target_col = "RUL"

df_test[feature_cols] = scaler_x.transform(df_test[feature_cols])
df_test[[target_col]] = scaler_y.transform(df_test[[target_col]])

# ---------------- Create sequences ---------------- #
sequence_length = 50
sequences, labels = create_sequences(df_test, sequence_length=sequence_length)
print(f"🧩 Total sequences in test set: {len(sequences)}")

X_test = np.array(sequences)
y_test = np.array(labels).reshape(-1, 1)

# ---------------- Load model ---------------- #
print("🧠 Loading trained LSTM model...")
model = load_model("models/lstm_fd002.h5", compile=False)

# ---------------- Predict ---------------- #
print("🔮 Generating predictions...")
y_pred_scaled = model.predict(X_test, batch_size=64, verbose=0)
y_pred = scaler_y.inverse_transform(y_pred_scaled)
y_true = scaler_y.inverse_transform(y_test)

# ---------------- Evaluate ---------------- #
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
mae = mean_absolute_error(y_true, y_pred)
r2 = r2_score(y_true, y_pred)

print(f"📊 Evaluation Metrics on test set -> RMSE: {rmse:.2f}, MAE: {mae:.2f}, R²: {r2:.2f}")

# ---------------- Plot predictions ---------------- #
plt.figure(figsize=(12, 5))
plt.plot(y_true, label="Actual RUL", linewidth=2)
plt.plot(y_pred, label="Predicted RUL", linestyle="--", linewidth=2)
plt.xlabel("Sample Index")
plt.ylabel("RUL")
plt.title("Remaining Useful Life (RUL) Prediction")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
