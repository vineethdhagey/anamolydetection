# Predictive Maintenance with NASA Turbofan FD002

This project implements a modular, MLOps-ready pipeline for predictive maintenance using the NASA Turbofan Engine Degradation Simulation Dataset (FD002). It includes real-time simulation, anomaly detection, and Remaining Useful Life (RUL) prediction using LSTM/GRU models.

---

## 🚀 Features

- **Data Ingestion**: Load FD002 from Hugging Face with optional real-time simulation
- **Preprocessing**: Normalization, rolling features, delta computation, sequence generation
- **Anomaly Detection**: Isolation Forest-based detection and visualization
- **RUL Prediction**: LSTM-based regression with RMSE/MAE evaluation
- **Streaming Inference**: Simulate sensor logs for real-time prediction
- **Dashboard**: Optional Streamlit/Plotly dashboard for live monitoring
- **MLOps-Ready**: Modular structure, logging, model saving, and config-driven workflows

---

## 📁 Project Structure
```
anamolydetection/
├── data/                      # Raw and processed data
│   ├── raw/
│   └── processed/
├── notebooks/                # EDA, prototyping
├── src/                      # Core logic
│   ├── preprocessing.py
│   ├── anomaly_detection.py
│   ├── rul_prediction.py
│   └── utils.py
├── models/                   # Saved models
├── scripts/                  # CLI scripts
│   ├── simulate_stream.py
│   ├── train_model.py
│   ├── evaluate_model.py
│   └── predict_rul.py
├── dashboards/               # Streamlit/Plotly dashboards
├── logs/                     # Logging outputs
├── config/                   # YAML/JSON config files
├── requirements.txt
└── README.md
└── .gitignore             # Ignore virtualenv, data, __pycache__, etc.

```

