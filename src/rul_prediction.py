import torch
import torch.nn as nn

class RULPredictor(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers,
                            batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        return self.fc(hn[-1])


def train_model(model, train_loader, val_loader=None, epochs=30):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            output = model(x_batch)
            loss = criterion(output.squeeze(), y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs}, Avg Loss: {avg_loss:.6f}")

        if val_loader:
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for x_val, y_val in val_loader:
                    y_pred = model(x_val)
                    val_loss += criterion(y_pred.squeeze(), y_val).item()
            val_loss /= len(val_loader)
            print(f"  Validation Loss: {val_loss:.6f}")
