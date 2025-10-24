from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt

def detect_anomalies(df, sensor_cols):
    if len(df) == 0:
        df['anomaly'] = []
        return df, None

    model = IsolationForest(contamination=0.01, random_state=42)
    df['anomaly'] = model.fit_predict(df[sensor_cols])
    return df, model


def plot_anomalies(df, unit=1, sensor='s_2'):
    unit_df = df[df['unit_number'] == unit]
    plt.figure(figsize=(12, 4))
    plt.plot(unit_df['time_cycles'], unit_df[sensor], label=sensor)
    plt.scatter(unit_df['time_cycles'][unit_df['anomaly'] == -1],
                unit_df[sensor][unit_df['anomaly'] == -1],
                color='red', label='Anomaly')
    plt.legend()
    plt.title(f'Anomalies in {sensor} for Unit {unit}')
    plt.show()
