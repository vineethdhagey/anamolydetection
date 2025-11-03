from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt

def detect_anomalies(df, sensor_cols):
    if len(df) == 0:
        df['anomaly'] = []
        return df, None

    model = IsolationForest(contamination=0.01, random_state=42)
    df['anomaly'] = model.fit_predict(df[sensor_cols])
    return df, model

class Autoencoder(nn.Module):
    def __init__(self, input_dim, encoding_dim=32):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, encoding_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

def train_autoencoder(data, epochs=50, batch_size=32, learning_rate=0.001):
    input_dim = data.shape[1]
    model = Autoencoder(input_dim)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    dataset = TensorDataset(torch.tensor(data, dtype=torch.float32))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        for batch in dataloader:
            inputs = batch[0]
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()

    return model

def detect_anomalies_autoencoder(df, sensor_cols, model, threshold=None):
    if len(df) == 0:
        df['anomaly'] = []
        return df

    data = df[sensor_cols].values
    data_tensor = torch.tensor(data, dtype=torch.float32)
    with torch.no_grad():
        reconstructed = model(data_tensor)
        reconstruction_error = torch.mean((data_tensor - reconstructed) ** 2, dim=1).numpy()

    if threshold is None:
        threshold = np.percentile(reconstruction_error, 95)  # Default threshold at 95th percentile

    df['anomaly'] = (reconstruction_error > threshold).astype(int)
    df['reconstruction_error'] = reconstruction_error
    return df



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
