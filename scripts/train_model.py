import pandas as pd
import numpy as np
from datasets import load_dataset
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
import joblib
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.preprocessing import preprocess, create_sequences, calculate_rul


def apply_scalers(df, scalers):
    scaler_x, scaler_y = scalers
    df = calculate_rul(df)
    feature_cols = [c for c in df.columns if c not in ["RUL", "unit_number", "time_cycles"]]
    df[feature_cols] = scaler_x.transform(df[feature_cols])
    df[["RUL"]] = scaler_y.transform(df[["RUL"]])
    return df


# ----------------------------- LOAD DATA ----------------------------- #
train_raw = load_dataset("LucasThil/nasa_turbofan_degradation_FD002", split="train")
test_raw = load_dataset("LucasThil/nasa_turbofan_degradation_FD002", split="valid")

train_df = pd.DataFrame(train_raw)
test_df = pd.DataFrame(test_raw)

# ----------------------------- PREPROCESS ----------------------------- #
train_df, (scaler_x, scaler_y) = preprocess(train_df)  # fit scalers on train only
test_df = apply_scalers(test_df, (scaler_x, scaler_y))  # apply to test only

# ----------------------------- SEQUENCES ----------------------------- #
X_train, y_train = create_sequences(train_df, sequence_length=50)
X_test, y_test = create_sequences(test_df, sequence_length=50)

X_train, y_train = np.array(X_train), np.array(y_train)
X_test, y_test = np.array(X_test), np.array(y_test)

print("✅ Data ready:")
print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)

# ----------------------------- MODEL ----------------------------- #
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dropout(0.2),
    LSTM(32),
    Dropout(0.2),
    Dense(1)
])

model.compile(optimizer="adam", loss="mse")

# ----------------------------- TRAIN ----------------------------- #
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=30,
    batch_size=64
)

# ----------------------------- SAVE ----------------------------- #
model.save("models/lstm_fd002.h5")
joblib.dump(scaler_x, "models/feature_scaler.pkl")
joblib.dump(scaler_y, "models/rul_scaler.pkl")

print("✅ Model and scalers saved successfully.")
