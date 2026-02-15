import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from train import get_session_vector  # your session encoder
import torch
import pandas as pd
import numpy as np
import torch.nn.functional as F


class AttentionLayer(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.attn = nn.Linear(input_dim, 1)

    def forward(self, x):  # x: [B, T, D]
        weights = self.attn(x).squeeze(-1)  # [B, T]
        weights = F.softmax(weights, dim=1)  # [B, T]
        weighted = torch.bmm(weights.unsqueeze(1), x)  # [B, 1, D]
        return weighted.squeeze(1)  # [B, D]

class EdgeRefinerNet(nn.Module):
    def __init__(self, input_dim=14, hidden_size=256):
        super().__init__()
        self.cnn = nn.Conv1d(in_channels=input_dim, out_channels=128, kernel_size=5, padding=2)

        self.lstm = nn.LSTM(input_size=128, hidden_size=hidden_size, batch_first=True, bidirectional=True)
        self.attn = AttentionLayer(hidden_size * 2)

        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 2)  # [refined_up, refined_down]
        )

    def forward(self, x):  # x: [B, T, D]
        x = x.transpose(1, 2)  # [B, D, T] for CNN
        x = self.cnn(x)        # [B, C, T]
        x = x.transpose(1, 2)  # [B, T, C] for LSTM

        lstm_out, _ = self.lstm(x)  # [B, T, 2H]
        context = self.attn(lstm_out)  # [B, 2H]
        out = self.fc(context)  # [B, 2]
        return out

def prepare_meta_data(df, base_preds, edge_preds, seq_len=10):
    X_meta = []
    y_meta = []

    for i in range(130 + seq_len - 1, len(df) - 10):
        sequence = []
        for j in range(i - seq_len + 1, i + 1):  # last `seq_len` candles
            candle = df.iloc[j]
            up_pred, down_pred = base_preds[j]
            edge = edge_preds[j]
            session_vec = get_session_vector(candle.name.hour)

            open_, high, low, close, vol = candle[["Open", "High", "Low", "Close", "Volume"]]
            atr = candle["ATR"]
            ema = candle["EMA20"]

            features = [open_, high, low, close, vol, atr, ema, edge, up_pred, down_pred] + list(session_vec)
            sequence.append(features)

        future = df.iloc[i+1:i+11]
        entry = df.iloc[i]["Open"]
        high_future = future["High"].max()
        low_future = future["Low"].min()

        true_up = (high_future - entry) / entry * 100
        true_down = (entry - low_future) / entry * 100

        X_meta.append(sequence)
        y_meta.append([true_up, true_down])

    return np.array(X_meta), np.array(y_meta)


def train_refiner(X, y, epochs=120, lr=1e-3):
    model = EdgeRefinerNet(input_dim=X.shape[2])
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-5
    )

    dataset = TensorDataset(torch.tensor(X, dtype=torch.float32),
                            torch.tensor(y, dtype=torch.float32))
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for xb, yb in loader:
            pred = model(xb)
            loss = loss_fn(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        scheduler.step(avg_loss)  # adapt learning rate

        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}")

    return model


def build_meta_input(candle, up_pred, down_pred, edge):
    open_, high, low, close, vol = candle[["Open", "High", "Low", "Close", "Volume"]]
    atr = candle["ATR"]
    ema = candle["EMA20"]
    session_vec = get_session_vector(candle.name.hour)

    features = [open_, high, low, close, vol, atr, ema, edge, up_pred, down_pred] + list(session_vec)
    return torch.tensor(features, dtype=torch.float32).unsqueeze(0)




