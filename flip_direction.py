import numpy as np
import pandas as pd
from scipy.signal import hilbert
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader

def label_flip(idx, prices, timestamps, next_index=None):
    open_prices = prices[:, 0]
    high_prices = prices[:, 1]
    low_prices = prices[:, 2]

    entry_price = open_prices[idx]
    next_idx = next_index if next_index is not None else len(prices)

    high_window = high_prices[idx + 1:next_idx]
    low_window = low_prices[idx + 1:next_idx]

    max_up = np.max(high_window) if len(high_window) > 0 else entry_price
    max_down = np.min(low_window) if len(low_window) > 0 else entry_price

    up_range = (max_up - entry_price) / entry_price
    down_range = (entry_price - max_down) / entry_price

    return [up_range, down_range]


def build_direction_dataset(flips, prices, timestamps, up, down):
    open_ = prices[:, 0]
    high = prices[:, 1]
    low = prices[:, 2]
    close = prices[:, 3]
    volume = prices[:, 4]

    signal = up - down
    direction = np.real(hilbert(up / (-down)))
    direction[direction < -9] = 9
    momentum = execution_generator(signal)

    delta_signal = np.diff(signal, prepend=signal[0])
    delta_direction = np.diff(direction, prepend=direction[0])
    delta_volume = np.diff(volume, prepend=volume[0])

    X, y = [], []

    for i, idx in enumerate(flips):
        if idx < 4 or idx >= len(prices) - 2:
            continue

        # === Feature window (5 timesteps, 11 features) ===
        ohlcv_window = np.stack([
            open_[idx-4:idx+1],
            high[idx-4:idx+1],
            low[idx-4:idx+1],
            close[idx-4:idx+1],
            volume[idx-4:idx+1]
        ], axis=1)

        feature_window = np.stack([
            signal[idx-4:idx+1],
            direction[idx-4:idx+1],
            momentum[idx-4:idx+1],
            delta_signal[idx-4:idx+1],
            delta_direction[idx-4:idx+1],
            delta_volume[idx-4:idx+1]
        ], axis=1)

        combined = np.concatenate([ohlcv_window, feature_window], axis=1)

        # === Labeling ===
        next_idx = flips[i + 1] if i + 1 < len(flips) else None
        up_range, down_range = label_flip(idx, prices, timestamps, next_index=next_idx)

        if up_range > 0.002 and down_range < 0.0015:
            label = 1  # LONG
        elif down_range > 0.002 and up_range < 0.0015:
            label = 0  # SHORT
        else:
            continue  # skip low-confidence flips

        X.append(combined)
        y.append(label)

    print(f"Samples: {len(X)} | LONG: {y.count(1)} | SHORT: {y.count(0)}")
    return np.array(X), np.array(y, dtype=np.int64)

def execution_generator(direction):
    execution = [0]*len(direction)
    for i in range(len(execution)):
        integral = 0
        if i >= 4:
            for j in range(4):
                integral = integral + direction[i-j]
        else:
            for j in range(i):
                integral = integral + direction[i-j]
        execution[i] = integral
    return execution

# ================================================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F

class TemporalAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Conv1d(channels, 1, kernel_size=1),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        weights = self.attn(x)  # shape: (B, 1, T)
        return x * weights

class FlipDirectionNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(11, 128, 3, padding=1)
        self.bn1 = nn.BatchNorm1d(128)
        self.dropout1 = nn.Dropout(0.2)

        self.conv2 = nn.Conv1d(128, 256, 3, padding=1)
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout2 = nn.Dropout(0.2)

        self.attn = TemporalAttention(256)

        self.bilstm = nn.LSTM(input_size=256, hidden_size=128, batch_first=True, bidirectional=True)

        self.fc = nn.Linear(256, 2)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.dropout1(x)

        x = F.relu(self.bn2(self.conv2(x)))
        x = self.dropout2(x)

        x = self.attn(x)

        x = x.permute(0, 2, 1)  # (B, T, C)
        x, _ = self.bilstm(x)
        x = x[:, -1, :]  # take final timestep

        return self.fc(x)
    
def train_model(X, y, epochs=20, batch_size=64, lr=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y)

    train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                             torch.tensor(y_train, dtype=torch.long))
    val_ds = TensorDataset(torch.tensor(X_val, dtype=torch.float32),
                           torch.tensor(y_val, dtype=torch.long))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    model = FlipDirectionNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                out = model(xb)
                val_loss += criterion(out, yb).item()
                preds = out.argmax(dim=1)
                correct += (preds == yb).sum().item()
                total += yb.size(0)

        acc = correct / total
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Acc: {acc:.4f}")

    return model