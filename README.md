# Predictive Maintenance with NASA Turbofan FD002

A comprehensive, MLOps-ready pipeline for predictive maintenance using the NASA Turbofan Engine Degradation Simulation Dataset (FD002). This project implements real-time anomaly detection, Remaining Useful Life (RUL) prediction, and interactive visualization to predict engine failures before they occur.

## 🚀 Features

- **Data Ingestion**: Automated loading of FD002 dataset from Hugging Face Datasets
- **Preprocessing Pipeline**: Feature scaling, RUL calculation with capping, sequence generation for time-series modeling
- **Anomaly Detection**: Isolation Forest-based detection with visualization capabilities
- **RUL Prediction**: LSTM implementations (Keras and PyTorch) for regression-based RUL estimation
- **Real-time Simulation**: Streaming data simulation for real-time inference scenarios
- **Interactive Dashboard**: Streamlit-based web app with Plotly visualizations for live monitoring
- **MLOps-Ready**: Modular architecture, model serialization, logging, and configuration-driven workflows
- **Scalable Architecture**: Supports batch processing and streaming inference

## 📊 Dataset

**NASA Turbofan FD002** is a multivariate time-series dataset simulating turbofan engine degradation under variable operating conditions. Key characteristics:

- **Units**: 260 engines (train) + validation/test sets
- **Features**: 3 operational settings + 21 sensor measurements
- **Time Series**: Variable-length sequences per engine (cycles until failure)
- **Target**: Remaining Useful Life (RUL) prediction
- **Source**: [Hugging Face Datasets](https://huggingface.co/datasets/LucasThil/nasa_turbofan_degradation_FD002)

## 🏗️ Architecture & Workflow

### Core Components

1. **Data Layer** (`src/utils.py`):
   - Dataset loading from Hugging Face
   - Streaming simulation generator

2. **Preprocessing Layer** (`src/preprocessing.py`):
   - RUL calculation with configurable capping (default: 125 cycles)
   - MinMax scaling for features and targets
   - Sequence creation for LSTM input (sliding windows)

3. **Anomaly Detection** (`src/anomaly_detection.py`):
   - Isolation Forest with contamination tuning
   - Visualization of anomalies over time cycles

4. **RUL Prediction** (`src/rul_prediction.py`):
   - PyTorch LSTM model with configurable layers
   - Training loop with validation
   - Keras alternative in training scripts



### Workflow Pipeline

```
Raw Data → Preprocessing → Feature Engineering → Model Training → Deployment
     ↓           ↓              ↓                    ↓              ↓
  FD002    Scaling/RUL    Sequences/         LSTM/Keras           Dashboard
  Load     Calculation   Normalization       Training             Inference
```




## Architecture Workflow

                                                            ┌──────────────────────────┐
                                                            │  Data Ingestion          │
                                                            │ Load NASA FD002 (HF)     │
                                                            │ Train/val/test frames    │
                                                            └──────────┬───────────────┘
                                                                    ▼
                                                            ┌──────────────────────────┐
                                                            │    Preprocessing + Seq   │
                                                            │      MinMax scaling      │
                                                            │                          │
                                                            │ Sliding-window sequences │
                                                            └──────────┬───────────────┘
                                                                    ▼
                                                            ┌──────────────────────────┐
                                                            │  Anomaly Detection       │
                                                            │  Isolation Forest        │
                                                            │ Mark/visualize anomalies │
                                                            └──────────┬───────────────┘
                                                                    ▼
                                                            ┌──────────────────────────┐
                                                            │    Model Training        │
                                                            │        LSTM              │
                                                            │ Save models (.pt/.h5)    │
                                                            │ Save scalers (.pkl)      │
                                                            └──────────┬───────────────┘
                                                                    ▼
                                                            ┌──────────────────────────┐
                                                            │    Evaluation            │
                                                            │                          │
                                                            │ Compare Pred RUL vs True │
                                                            │ Plot predictions         │
                                                            └──────────┬───────────────┘
                                                                    ▼
                                                            ┌──────────────────────────┐
                                                            │    Deployment            │
                                                            │ Batch + Streaming infer  │
                                                            │ Realtime sim + dashboard │
                                                            │ (Streamlit monitoring)   │
                                                            └──────────────────────────┘





















## 📁 Project Structure

```
predictive_maintenance/
├── data/
│   ├── raw/                 # Raw dataset storage (auto-downloaded)
│   └── processed/           # Preprocessed data (sequences, scaled features)
├── notebooks/
│   └── note.ipynb           # Jupyter notebook for prototyping and EDA
├── src/                     # Core source code
│   ├── __init__.py
│   ├── utils.py             # Data loading and streaming utilities
│   ├── preprocessing.py     # Feature engineering and sequence creation
│   ├── anomaly_detection.py # Isolation Forest implementation
│   ├── rul_prediction.py    # PyTorch LSTM model and training
│   └── evaluation.py        # Metrics and visualization functions
├── scripts/                 # Executable scripts
│   ├── __init__.py
│   ├── train_model.py       # Keras LSTM training pipeline
│   ├── evaluate_model.py    # Model evaluation on validation set
│   ├── predict_rul.py       # Batch RUL prediction on validation data
│   └── simulate_stream.py   # Streaming data simulation
├── dashboards/
│   └── app.py               # Streamlit dashboard for interactive monitoring
├── models/                  # Saved models and scalers
│   ├── rul_model.pt         # PyTorch LSTM model
│   ├── lstm_fd002.h5        # Keras LSTM model
│   ├── lstm_fd002_best.h5   # Best Keras model checkpoint
│   ├── feature_scaler.pkl   # Feature scaler
│   ├── rul_scaler.pkl       # Target scaler
│   └── *.pkl                # Additional scaler variants
├── logs/                    # Training logs and runtime outputs
├── config/                  # YAML/JSON configuration files (extensible)
├── requirements.txt         # Python dependencies
├── .gitignore               # Git ignore rules
└── README.md                # This file
```

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- pip package manager
- Virtual environment (recommended)

### Setup Steps

1. **Clone the repository**:
   ```bash
   git clone https://github.com/vineethdhagey/anamolydetection.git
   cd predictive_maintenance
   ```

2. **Create virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify installation**:
   ```bash
   python -c "import torch, pandas, sklearn; print('Dependencies installed successfully')"
   ```

## 🚀 Usage

### 1. Data Preparation
```bash
# Load and preprocess data (via notebook or scripts)
python -c "from src.utils import load_fd002_dataset; df = load_fd002_dataset('train')"
```

### 2. Model Training
```bash
# Train Keras LSTM model
python scripts/train_model.py

# Or use PyTorch (from notebook)
# Follow notebooks/note.ipynb for PyTorch training
```



### 3. Batch Prediction
```bash
# Predict RUL on validation data
python scripts/predict_rul.py
```

### 4. Streaming Simulation
```bash
# Simulate real-time data streaming
python scripts/simulate_stream.py
```

### 5. Launch Dashboard
```bash
# Start interactive dashboard
streamlit run dashboards/app.py
```

## 📈 Model Details

### LSTM Architecture
- **Input**: Sequences of 50 time cycles × 24 features (3 settings + 21 sensors)
- **Layers**: 2 LSTM layers (128 → 64 units) + Dense output
- **Regularization**: Dropout (0.2), Batch Normalization
- **Optimizer**: Adam (lr=0.001)
- **Loss**: MSE with MAE metric

### Training Configuration
- **Batch Size**: 64
- **Epochs**: 100 (with early stopping)
- **Validation Split**: 20%
- **Sequence Length**: 50 cycles
- **RUL Cap**: 125 cycles

### Evaluation Metrics
- **RMSE**: Root Mean Squared Error
- **MAE**: Mean Absolute Error
- **R² Score**: Coefficient of determination

## 🎯 Dashboard Features

The Streamlit dashboard provides:

- **Unit Selection**: Dropdown for engine unit selection
- **Sensor Visualization**: Time-series plots with anomaly markers
- **RUL Prediction**: Smoothed prediction curves
- **Interactive Controls**: Toggle anomaly detection and RUL prediction
- **Raw Data Viewer**: Expandable data table

### Dashboard Screenshots
<img width="1683" height="837" alt="rul prediction" src="https://github.com/user-attachments/assets/47c0aa27-b4b2-42e0-ae70-c4e7f050a849" />




<img width="1520" height="770" alt="rul-2" src="https://github.com/user-attachments/assets/9596e99a-f5d1-4442-9770-9c9ceae305c9" />

*Real-time monitoring interface showing sensor data, anomalies, and RUL predictions*

## 🔧 Configuration

The project supports configuration-driven workflows. Add YAML/JSON files to `config/` for:

- Model hyperparameters
- Dataset parameters
- Training settings
- Evaluation thresholds

Example config structure:
```yaml
model:
  sequence_length: 50
  batch_size: 64
  epochs: 100
  learning_rate: 0.001

data:
  rul_cap: 125
  contamination: 0.01
```

## 📦 Dependencies

Key packages (see `requirements.txt`):

- **Data Processing**: `pandas`, `numpy`
- **Machine Learning**: `scikit-learn`, `torch`, `tensorflow`
- **Visualization**: `matplotlib`, `plotly`
- **Data Loading**: `datasets` (Hugging Face)
- **Dashboard**: `streamlit`
- **Serialization**: `joblib`

## 🧪 Testing & Validation

### Performance Validation
- Cross-validation on training set
- Real-time simulation testing

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request




## 📞 Contact

This project was developed by : **Vineeth Dhagey**,


For questions or collaborations, please open an issue or reach out to me.

---

**Note**: This project demonstrates end-to-end MLOps practices for predictive maintenance, suitable for production deployment with additional monitoring and CI/CD integration.
