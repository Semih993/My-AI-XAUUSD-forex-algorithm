import pandas as pd
import requests
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import torch.nn.functional as F
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import ta
from datetime import timedelta, datetime, timezone
from typing import Tuple
from FVG import add_fvg_features
from wick_liquidity import wick_liquidity

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ========== FETCH FOREX DATA (EUR/USD, OANDA) ==========
def fetch_xau_data(pair="XAU_USD", granularity="M30", price_type="M", start_date="live", max_candles=5000):
    """
    Fetches XAU/USD candles from OANDA.

    - If start_date == 'live', fetches the latest `max_candles` candles.
    - Otherwise, fetches candles starting from `start_date`.

    Args:
        pair (str): OANDA instrument (default: "XAU_USD")
        granularity (str): Candle granularity (e.g., "M1", "M30", "H1")
        price_type (str): "M" (mid), "B" (bid), or "A" (ask)
        start_date (str or datetime): "live" or starting datetime in UTC
        max_candles (int): Number of candles to fetch

    Returns:
        pd.DataFrame: OHLCV data indexed by datetime
    """

    # 🔐 Hardcoded API key as requested
    api_key = "9fda30eddf5312cf7c53896c8184f814-6dec5d706fb205e81a57d51f9a2a1d94"
    price_field_map = {
        "M": "mid",
        "B": "bid",
        "A": "ask"
    }
    assert price_type in price_field_map, "price_type must be 'M', 'B', or 'A'"
    field = price_field_map[price_type]

    headers = {
        "Authorization": f"Bearer {api_key}"
    }

    base_url = f"https://api-fxpractice.oanda.com/v3/instruments/{pair}/candles"

    if start_date == "live":
        # ✅ Original behavior: fetch last max_candles candles from now
        params = {
            "granularity": granularity,
            "count": max_candles,
            "price": price_type
        }

        response = requests.get(base_url, headers=headers, params=params)
        if response.status_code != 200:
            raise Exception(f"Failed to fetch data: {response.text}")

        candles = response.json()["candles"]
        candles = [c for c in candles if c["complete"]]

    else:
        # ✅ New behavior: fetch from given date using pagination
        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date)
        if not isinstance(start_date, datetime):
            raise ValueError("start_date must be a string or datetime object")

        start_iso = start_date.replace(tzinfo=timezone.utc).isoformat()
        candles = []
        next_time = start_iso
        remaining = max_candles

        while remaining > 0:
            count = min(500, remaining)
            params = {
                "granularity": granularity,
                "price": price_type,
                "count": count,
                "from": next_time
            }

            response = requests.get(base_url, headers=headers, params=params)
            if response.status_code != 200:
                raise Exception(f"Failed to fetch data: {response.text}")

            batch = response.json().get("candles", [])
            batch = [c for c in batch if c["complete"]]
            if not batch:
                break

            candles.extend(batch)
            next_time = batch[-1]["time"]
            remaining -= len(batch)

            if len(batch) < count:
                break

    # === Parse candles ===
    ohlcv = [
        [
            c["time"],
            float(c[field]["o"]),
            float(c[field]["h"]),
            float(c[field]["l"]),
            float(c[field]["c"]),
            float(c["volume"])
        ]
        for c in candles
    ]

    df = pd.DataFrame(ohlcv, columns=["Date", "Open", "High", "Low", "Close", "Volume"])
    df["Date"] = pd.to_datetime(df["Date"])
    df.set_index("Date", inplace=True)
    return df

def apply_and_gate_smoothing(X: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Applies a binary AND gate to the y labels using a window size of 2.
    Only keeps a trade (1 or -1) if both it and the next label are equal.
    If they disagree or include a 0 (no-trade), the label is set to 0.

    Returns the adjusted (X, y) tensors, shortened by 1 sample.
    """
    assert len(X) == len(y), "Mismatch between X and y length"

    new_X = []
    new_y = []

    for i in range(len(y) - 1):
        a = y[i].item()
        b = y[i + 1].item()
        if a == b and a in [1, -1]:  # Both are 1 or both are -1
            new_X.append(X[i])
            new_y.append(a)
        else:
            new_X.append(X[i])
            new_y.append(0)  # Force to no-trade

    return torch.stack(new_X), torch.tensor(new_y, dtype=torch.float32)

# ========== SESSION ENCODING (UK TIME) ==========
def get_session_vector(hour):
    if 22 <= hour or hour < 0:
        return [100, 0, 0, 0]  # Sydney
    elif 0 <= hour < 8:
        return [0, 100, 0, 0]  # Asian
    elif 8 <= hour < 13:
        return [0, 0, 100, 0]  # London
    elif 13 <= hour < 22:
        return [0, 0, 0, 100]  # New York
    else:
        return [0, 0, 0, 0]  # Fallback
    
# ========== COMPUTE ATR ============
def compute_atr(df, period=14):
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    atr = tr.rolling(window=period).mean()
    return atr

# ========== PREPARE DATA ==========
def prepare_data(
    df: pd.DataFrame,
    lookback: int = 30,
    lookahead: int = 10,
    min_wick_range: float = 0.0,
    debug_time: str = None  # Optional: e.g., "2024-08-06 17:00:00+00:00"
):
    """
    Prepares model input X and continuous labels y_up and y_down.

    Returns:
        X: [samples, lookback, features]
        y_up: expected up % move
        y_down: expected down % move
        times: list of timestamps
    """

    # === Technical Indicators ===
    df["EMA20"] = ta.trend.EMAIndicator(df["Close"], window=20).ema_indicator()
    df["OBV"] = ta.volume.OnBalanceVolumeIndicator(df["Close"], df["Volume"]).on_balance_volume()
    df["BodySize"] = (df["Close"] - df["Open"]).abs()
    df["Range"] = (df["High"] - df["Low"]).abs()

    # === ATR ===
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df["ATR"] = tr.rolling(window=14).mean()

    # === Structure and wick features ===
    df = add_fvg_features(df)
    df = wick_liquidity(df)
    df.dropna(inplace=True)  # 🔥 Important: drop before labeling

    # === Label generation ===
    df = generate_up_down_labels(df, lookahead=lookahead)

    # === Session encoding ===
    session_vecs = np.array([get_session_vector(ts.hour) for ts in df.index])

    # === Feature matrix ===
    base_cols = ["Open", "High", "Low", "Close", "Volume", "Range", "ATR"]
    fvg_cols = [col for col in df.columns if col.startswith("FVG_")]
    tech_cols = ["EMA20", "OBV", "BodySize"]
    feature_data = df[base_cols + fvg_cols + tech_cols]

    # === Normalize and stack session encoding ===
    scaler = MinMaxScaler(feature_range=(0, 100))
    scaled = scaler.fit_transform(feature_data)
    combined = np.hstack([scaled, session_vecs])

    combined_df = pd.DataFrame(
        combined,
        columns=feature_data.columns.tolist() + ["S1", "S2", "S3", "S4"],
        index=feature_data.index
    )

    highs = df["High"].values
    lows = df["Low"].values
    closes = df["Close"].values
    up_labels = df["ExpectedUp"].values
    down_labels = df["ExpectedDown"].values

    X, y_up, y_down, times = [], [], [], []

    for i in range(len(combined_df) - lookback - lookahead):
        wick_range = max(highs[i+lookback : i+lookback+lookahead]) - min(lows[i+lookback : i+lookback+lookahead])
        if wick_range < min_wick_range:
            continue

        X.append(combined_df.iloc[i:i+lookback].values)
        y_up.append(up_labels[i + lookback - 1])
        y_down.append(down_labels[i + lookback - 1])
        times.append(combined_df.index[i + lookback - 1])

    # === Debugging: verify true label from source ===
    if debug_time:
        try:
            dt = pd.to_datetime(debug_time)
            idx = df.index.get_loc(dt)
            entry_price = closes[idx]
            window = df.iloc[idx+1 : idx+1+lookahead]
            true_up = (window["High"].max() - entry_price) / entry_price * 100
            true_down = (entry_price - window["Low"].min()) / entry_price * 100
            print(f"\n[DEBUG] {debug_time} entry: {entry_price:.2f}")
            print(f"[DEBUG] True max up over next {lookahead}: {true_up:.3f}%")
            print(f"[DEBUG] True max down over next {lookahead}: {true_down:.3f}%")
            print(f"[DEBUG] Stored label: ↑ {df.loc[dt, 'ExpectedUp']}%, ↓ {df.loc[dt, 'ExpectedDown']}%")
        except Exception as e:
            print(f"[DEBUG] Could not evaluate debug time: {e}")

    return (
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y_up, dtype=torch.float32),
        torch.tensor(y_down, dtype=torch.float32),
        times
    )


# ============================== Generate Up-Down Labels =====================================
def generate_up_down_labels(df: pd.DataFrame, lookahead: int = 10) -> pd.DataFrame:
    """
    Adds 'ExpectedUp' and 'ExpectedDown' columns to the dataframe.
    These indicate the maximum percentage movement up and down from the close
    over the next `lookahead` candles.

    Args:
        df (pd.DataFrame): DataFrame with OHLCV including 'High', 'Low', and 'Close'
        lookahead (int): Number of future candles to evaluate

    Returns:
        pd.DataFrame with added 'ExpectedUp' and 'ExpectedDown' columns
    """
    closes = df["Close"].values
    highs = df["High"].values
    lows = df["Low"].values

    up_moves = []
    down_moves = []

    for i in range(len(df)):
        if df.index[i] == pd.Timestamp("2024-08-06 16:00:00+00:00"):
            print(f"\n[🔍 VERIFY] Index: {i}")
            #print(f"[🔍] Entry price: {closes[i]:.2f}")
            #print(f"[🔍] Future high window: {highs[i+1:i+1+20]}")
            #print(f"[🔍] Future low window : {lows[i+1:i+1+20]}")
            #print(f"[🔍] Max high : {future_high:.2f}")
            #print(f"[🔍] Min low  : {future_low:.2f}")
            #print(f"[🔍] Up %    : {up_pct:.3f}%")
            #print(f"[🔍] Down %  : {down_pct:.3f}%")
        if i + lookahead >= len(df):
            up_moves.append(0.0)
            down_moves.append(0.0)
            continue

        future_high = highs[i+1:i+1+lookahead].max()
        future_low = lows[i+1:i+1+lookahead].min()
        entry_price = closes[i]

        up_pct = (future_high - entry_price) / entry_price * 100
        down_pct = (entry_price - future_low) / entry_price * 100

        up_moves.append(round(up_pct, 3))
        down_moves.append(round(down_pct, 3))

    df["ExpectedUp"] = up_moves
    df["ExpectedDown"] = down_moves
    return df


# ========== CUSTOM DATASET ==========
class XAUDataset(Dataset):
    def __init__(self, X, y_up, y_down):
        self.X = X
        self.y_up = y_up
        self.y_down = y_down

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y_up[idx], self.y_down[idx]

# ========== CNN-LSTM MODEL ==========
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.act1 = nn.GELU()

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.act2 = nn.GELU()

        self.skip = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        identity = self.skip(x)
        out = self.act1(self.bn1(self.conv1(x)))
        out = self.act2(self.bn2(self.conv2(out)))
        return out + identity

class AttentionLayer(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attn = nn.Linear(hidden_size, 1)

    def forward(self, x):  # x: [B, T, H]
        scores = self.attn(x).squeeze(-1)            # [B, T]
        weights = torch.softmax(scores, dim=1)       # [B, T]
        context = torch.sum(weights.unsqueeze(-1) * x, dim=1)  # [B, H]
        return context

class CNNLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=512):
        super(CNNLSTM, self).__init__()

        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=512, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(512)
        self.act1 = nn.GELU()
        self.dropout1 = nn.Dropout(0)

        self.conv2 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(512)
        self.act2 = nn.GELU()
        self.dropout2 = nn.Dropout(0)

        self.lstm = nn.LSTM(input_size=512, hidden_size=hidden_size, batch_first=True)

        # Output: [ExpectedUp, ExpectedDown]
        self.fc = nn.Linear(hidden_size, 2)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # to [B, features, time]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.dropout1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act2(x)
        x = self.dropout2(x)

        x = x.permute(0, 2, 1)  # back to [B, time, features]
        x, _ = self.lstm(x)
        x = x[:, -1, :]  # last time step output

        return self.fc(x)  # [B, 2]

# === Custom loss to penalize overuse of no-trade class ===
def penalized_loss(logits, y_true, penalty_weight=0.3):
    """
    Combines CrossEntropyLoss with a penalty if the model predicts too many NO-TRADEs (class index 1).
    """
    ce = F.cross_entropy(logits, (y_true + 1).long())  # Shift to [0, 1, 2]
    preds = torch.argmax(logits, dim=1)
    no_trade_ratio = (preds == 1).float().mean()
    return ce + penalty_weight * no_trade_ratio

# ========================== Train Model ================================
def train_model(model, train_loader, epochs=70, lr=0.001):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for X_batch, y_up, y_down in train_loader:
            X_batch = X_batch.to(device)
            targets = torch.stack([y_up, y_down], dim=1).to(device)  # shape [B, 2]

            optimizer.zero_grad()
            outputs = model(X_batch)  # [B, 2]
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()
        print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss:.4f}")
        
# ========== TESTING ==========
def test_model(model, test_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for X_batch, y_up, y_down in test_loader:
            X_batch = X_batch.to(device)
            targets = torch.stack([y_up, y_down], dim=1).to(device)

            outputs = model(X_batch)  # [B, 2]
            all_preds.append(outputs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    # Stack all batches
    all_preds = np.vstack(all_preds)
    all_targets = np.vstack(all_targets)

    # Separate components
    pred_up, pred_down = all_preds[:, 0], all_preds[:, 1]
    true_up, true_down = all_targets[:, 0], all_targets[:, 1]

    # Metrics
    mae_up = mean_absolute_error(true_up, pred_up)
    mae_down = mean_absolute_error(true_down, pred_down)
    rmse_up = mean_squared_error(true_up, pred_up) ** 0.5
    rmse_down = mean_squared_error(true_down, pred_down) ** 0.5
    r2_up = r2_score(true_up, pred_up)
    r2_down = r2_score(true_down, pred_down)

    print("\n📊 Test Metrics:")
    print(f"↑ MAE: {mae_up:.3f} | RMSE: {rmse_up:.3f} | R²: {r2_up:.3f}")
    print(f"↓ MAE: {mae_down:.3f} | RMSE: {rmse_down:.3f} | R²: {r2_down:.3f}")

    return {
        "pred_up": pred_up,
        "pred_down": pred_down,
        "true_up": true_up,
        "true_down": true_down
    }