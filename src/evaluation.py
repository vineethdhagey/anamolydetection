import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

def evaluate(y_true, y_pred):
    """Compute and print regression evaluation metrics."""
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"📊 Evaluation Metrics -> RMSE: {rmse:.2f}, MAE: {mae:.2f}, R²: {r2:.3f}")

def plot_rul(y_true, y_pred):
    """Plot actual vs predicted Remaining Useful Life (RUL)."""
    plt.figure(figsize=(10, 5))
    plt.plot(y_true, label='Actual RUL', linewidth=2)
    plt.plot(y_pred, label='Predicted RUL', linestyle='--', linewidth=2)
    plt.xlabel('Sample Index')
    plt.ylabel('RUL')
    plt.title('🔧 Remaining Useful Life (RUL) Prediction')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
