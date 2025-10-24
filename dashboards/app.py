import sys
import os
import streamlit as st
import pandas as pd
import plotly.express as px
import torch
import joblib

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.utils import load_fd002_dataset
from src.preprocessing import preprocess, create_sequences
from src.anomaly_detection import detect_anomalies
from src.rul_prediction import RULPredictor

st.set_page_config(page_title="Predictive Maintenance Dashboard", layout="wide")

# ---------------- Sidebar ---------------- #
st.sidebar.title("Settings")

# ---------------- Load Test Data ---------------- #
@st.cache_data
def load_test_data():
    # Load validation/test split
    df_test = load_fd002_dataset(split="valid")  # FD002 has 'train' and 'valid'
    
    # Load scalers saved during training
    scaler_x = joblib.load("models/feature_scaler.pkl")
    scaler_y = joblib.load("models/rul_scaler.pkl")
    
    # Scale features using scaler_x
    feature_cols = ['setting_1', 'setting_2', 'setting_3'] + [f"s_{i}" for i in range(1, 22)]
    df_scaled = df_test.copy()
    df_scaled[feature_cols] = scaler_x.transform(df_test[feature_cols])
    
    return df_scaled, scaler_y

df, scaler_y = load_test_data()

# Unit selection dropdown (integer values)
unit_options = sorted(df['unit_number'].unique())
unit_id = st.sidebar.selectbox("Engine Unit ID", unit_options)

# Sensor selection
sensor = st.sidebar.selectbox("Sensor to visualize", [f"s_{i}" for i in range(1, 22)])

# Checkboxes
show_anomalies = st.sidebar.checkbox("Show Anomalies", value=True)
show_rul = st.sidebar.checkbox("Show RUL Prediction", value=True)

st.title("🚀 Predictive Maintenance Dashboard")
st.markdown("NASA Turbofan FD002 | Real-time anomaly detection and RUL prediction (Test Set)")

# Filter data for selected unit
unit_df = df[df['unit_number'] == unit_id].copy()

# ------------------ Anomaly Detection ------------------ #
if show_anomalies and len(unit_df) > 0:
    sensor_cols = [col for col in df.columns if col.startswith('s_')]
    unit_df, _ = detect_anomalies(unit_df, sensor_cols)
    fig = px.line(unit_df, x='time_cycles', y=sensor, title=f"{sensor} Over Time")
    anomalies = unit_df[unit_df['anomaly'] == -1]
    fig.add_scatter(
        x=anomalies['time_cycles'],
        y=anomalies[sensor],
        mode='markers',
        marker=dict(color='red', size=6),
        name='Anomaly'
    )
    st.plotly_chart(fig, use_container_width=True)
elif show_anomalies:
    st.warning(f"No data available for Unit {unit_id} to perform anomaly detection.")

# ------------------ RUL Prediction ------------------ #
if show_rul and len(unit_df) > 0:
    st.subheader("🔮 RUL Prediction")
    feature_cols = ['setting_1', 'setting_2', 'setting_3'] + [f"s_{i}" for i in range(1, 22)]
    sequence_df = unit_df[feature_cols + ['unit_number', 'time_cycles', 'RUL']].copy()
    sequences, _ = create_sequences(sequence_df, sequence_length=50)

    if sequences:
        X = torch.tensor(sequences, dtype=torch.float32)
        model = RULPredictor(input_size=X.shape[2])
        model.load_state_dict(torch.load("models/rul_model.pt", map_location=torch.device("cpu")))
        model.eval()

        with torch.no_grad():
            y_pred_scaled = model(X).numpy().flatten()
        y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1,1)).flatten()
        y_pred_smooth = pd.Series(y_pred).ewm(alpha=0.3).mean()

        fig2 = px.line(
            x=unit_df['time_cycles'][-len(y_pred_smooth):],
            y=y_pred_smooth,
            labels={'x': 'Time Cycles', 'y': 'RUL'},
            title=f"Predicted RUL for Unit {unit_id} (Smoothed)"
        )
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.warning(f"Not enough data points to create a sequence for Unit {unit_id}.")
else:
    if show_rul:
        st.warning(f"No data available for Unit {unit_id} to predict RUL.")

# ------------------ Raw Data ------------------ #
with st.expander("📄 View Raw Data"):
    st.dataframe(unit_df.head(50))
