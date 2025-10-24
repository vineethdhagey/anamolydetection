# scripts/predict_rul.py
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import pandas as pd
import numpy as np
import joblib
from src.preprocessing import preprocess, create_sequences
from src.rul_prediction import RULPredictor
from datasets import load_dataset

def fetch_validation_data():
    dataset = load_dataset("LucasThil/nasa_turbofan_degradation_FD002", split="valid")
    df = dataset.to_pandas()
    return df

def main():
    print("🌐 Fetching validation data from Hugging Face...")
    df = fetch_validation_data()

    print("🧹 Preprocessing...")
    df, _ = preprocess(df)

    print("🔗 Creating sequences...")
    sequences, _ = create_sequences(df, sequence_length=50)  # match training

    if len(sequences) == 0:
        print("❌ Not enough data to create sequences.")
        return

    print("📦 Preparing input tensors...")
    X = torch.tensor(np.array(sequences), dtype=torch.float32)

    print("🧠 Loading trained model...")
    model = RULPredictor(input_size=X.shape[2])
    model.load_state_dict(torch.load("models/rul_model.pt"))
    model.eval()

    # ------------------ Load scaler ------------------ #
    scaler_y = joblib.load("models/rul_scaler.pkl")  # must be saved during training

    print("🔮 Predicting RUL...")
    with torch.no_grad():
        y_pred_scaled = model(X).numpy().flatten()
        y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()

    print("📊 Predicted RULs (first 10):")
    for i, pred in enumerate(y_pred[:10]):
        print(f"Sample {i+1}: RUL = {pred:.2f}")

if __name__ == "__main__":
    main()
