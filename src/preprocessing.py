import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np

def calculate_rul(df, max_rul_cap=125):
    """
    Compute RUL per unit and cap at max_rul_cap.
    """
    df["RUL"] = df.groupby("unit_number")["time_cycles"].transform(lambda x: x.max() - x)
    df["RUL"] = df["RUL"].clip(upper=max_rul_cap)
    return df


def preprocess(df):
    """
    Preprocess FD002:
    - Sort by unit & cycles
    - Compute RUL + cap
    - Scale only sensor/setting features
    """
    df = df.sort_values(by=["unit_number", "time_cycles"]).reset_index(drop=True)

    # ✅ Compute RUL here (replaces previous capping logic)
    df = calculate_rul(df, max_rul_cap=125)

    # Features excluding identifiers & target
    feature_cols = [c for c in df.columns if c not in ["RUL", "unit_number", "time_cycles"]]
    target_col = "RUL"

    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()

    df[feature_cols] = scaler_x.fit_transform(df[feature_cols])
    df[[target_col]] = scaler_y.fit_transform(df[[target_col]])

    return df, (scaler_x, scaler_y)


def create_sequences(df, sequence_length=50):
    """
    Convert each unit's data into sequences for LSTM input.
    Returns sequences and corresponding RUL labels.
    """
    sequences, labels = [], []
    feature_cols = [c for c in df.columns if c not in ["unit_number", "time_cycles", "RUL"]]

    for unit in df["unit_number"].unique():
        unit_df = df[df["unit_number"] == unit]
        values = unit_df[feature_cols].values
        rul = unit_df["RUL"].values

        for i in range(len(values) - sequence_length):
            sequences.append(values[i:i+sequence_length])
            labels.append(rul[i+sequence_length])

    return sequences, labels
