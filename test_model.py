import pandas as pd
import torch
import numpy as np
from torch.utils.data import DataLoader
from datetime import datetime
import os
from termcolor import colored
from train import fetch_xau_data, prepare_data, get_session_vector, XAUDataset, CNNLSTM, train_model, test_model
from post_processing import CorrectionLSTM, train_correction_lstm_regression
from edge_prediction import EdgeRefinerNet, prepare_meta_data, train_refiner
import torch.nn.functional as F
import matplotlib.pyplot as plt


# ================= Outcome labeling ========================
def label_trade_outcome(entry_price, highs, lows, direction, tp=0.8, sl=0.53):
    if direction == 0:
        return 0
    target = entry_price * (1 + tp / 100) if direction == 1 else entry_price * (1 - tp / 100)
    stop = entry_price * (1 - sl / 100) if direction == 1 else entry_price * (1 + sl / 100)
    for hi, lo in zip(highs, lows):
        if direction == 1:
            if lo <= stop:
                return 0
            if hi >= target:
                return 1
        else:
            if hi >= stop:
                return 0
            if lo <= target:
                return 1
    return 0

# ========================== Refiner =============================#
def test_refiner(df, base_preds, edge_preds, model, seq_len=10):
    import torch
    from train import get_session_vector
    import pandas as pd

    model.eval()
    refined_preds = []

    offset = 130 + seq_len - 1

    for i in range(offset, len(df)):
        sequence = []
        for j in range(i - seq_len + 1, i + 1):
            candle = df.iloc[j]
            up_pred, down_pred = base_preds[j]
            edge = edge_preds[j]
            open_, high, low, close, vol = candle[["Open", "High", "Low", "Close", "Volume"]]
            atr = candle["ATR"]
            ema = candle["EMA20"]
            session_vec = get_session_vector(candle.name.hour)

            features = [open_, high, low, close, vol, atr, ema, edge, up_pred, down_pred] + list(session_vec)
            sequence.append(features)

        x_seq = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0)  # [1, T, F]

        with torch.no_grad():
            refined_up, refined_down = model(x_seq).squeeze().tolist()

        refined_preds.append((refined_up, refined_down))

    return refined_preds, offset

# ========================================== Main =============================================#
def main(date):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🎮 Using device: {device}")

    # === Load and prepare data ===

                                
    df = fetch_xau_data(start_date=date, granularity="M30", price_type="M")
    lookback, lookahead = 200, 5
    X, y_up, y_down, times = prepare_data(df, lookback=lookback, lookahead=lookahead)

    test_size = 4700
    split_idx = len(X) - test_size
    X_train, y_up_train, y_down_train = X[:split_idx], y_up[:split_idx], y_down[:split_idx]
    X_test, y_up_test, y_down_test = X[split_idx:], y_up[split_idx:], y_down[split_idx:]
    test_times = times[split_idx:]

    # === Train or Load CNN-LSTM model ===
    model = CNNLSTM(input_size=X.shape[2]).to(device)
    model_path = "trained_model.pth"
    LOAD_MAIN_MODEL = True
    if os.path.exists(model_path) and LOAD_MAIN_MODEL:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("📦 Loaded pretrained CNN-LSTM model")
    else:
        print("🔁 Training CNN-LSTM model...")
        train_loader = DataLoader(XAUDataset(X_train, y_up_train, y_down_train), batch_size=32)
        train_model(model, train_loader, epochs=110)
        torch.save(model.state_dict(), model_path)
        print("💾 Model saved")

    test_loader = DataLoader(XAUDataset(X_test, y_up_test, y_down_test), batch_size=32)
    results = test_model(model, test_loader)
    pred_up = np.array(results["pred_up"])
    pred_down = np.array(results["pred_down"])
    rr = pred_up / (pred_down + 1e-6)

    # =========== Correction LSTM ===========
    correction_model = CorrectionLSTM(input_size=5, hidden_size=128).to(device)
    corr_path = "correction_lstm.pth"
    seq_len = 10
    val_split = int(len(pred_up) * 0.7)
    LOAD_CORRECTION_MODEL = True

    if os.path.exists(corr_path) and LOAD_CORRECTION_MODEL:
        correction_model.load_state_dict(torch.load(corr_path, map_location=device))
        print("📦 Loaded pretrained Correction LSTM")
    else:
        print("🔁 Training Correction LSTM...")
        correction_model = train_correction_lstm_regression(
            pred_up[:val_split], pred_down[:val_split],
            results["true_up"][:val_split], results["true_down"][:val_split],
            device
        )
        torch.save(correction_model.state_dict(), corr_path)
        print("💾 Correction LSTM saved")

    ema20 = pd.Series(pred_up).rolling(20).mean().values
    atr = pd.Series(rr).rolling(14).mean().values
    delta_up = np.zeros_like(pred_up)
    delta_down = np.zeros_like(pred_down)
    count = np.zeros_like(pred_up)

    for i in range(val_split + seq_len, len(pred_up)):
        seq = np.stack([
            pred_up[i - seq_len:i],
            pred_down[i - seq_len:i],
            rr[i - seq_len:i],
            ema20[i - seq_len:i],
            atr[i - seq_len:i]
        ], axis=1)

        if np.isnan(seq).any(): continue

        seq_tensor = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(device)
        delta = correction_model(seq_tensor).cpu().detach().numpy().squeeze()
        for j in range(i - seq_len, i):
            delta_up[j] += delta[0]
            delta_down[j] += delta[1]
            count[j] += 1

    count = np.maximum(count, 1)
    corrected_up = pred_up + delta_up / count
    corrected_down = pred_down + delta_down / count
    edge = np.abs(corrected_up - corrected_down)

    # ============ Edge Refiner ===========
    ref_path = "refiner_model.pth"
    refiner = EdgeRefinerNet(input_dim=14, hidden_size=256)

    base_preds = list(zip(corrected_up, corrected_down))
    edge_preds = edge.tolist()

    LOAD_REFINER_MODEL = True
    if os.path.exists(ref_path) and LOAD_REFINER_MODEL:
        refiner.load_state_dict(torch.load(ref_path, map_location=device))
        print("📦 Loaded pretrained Refiner Model")
    else:
        X_meta, y_meta = prepare_meta_data(df.iloc[-len(base_preds):], base_preds, edge_preds)
        refiner = train_refiner(X_meta, y_meta)
        torch.save(refiner.state_dict(), "refiner_model.pth")
        print("✅ Successfully Trained Refiner Model")

    print("\n🔁 Applying Edge Refiner...")
    refined_preds, offset = test_refiner(df.iloc[-len(base_preds):], base_preds, edge_preds, refiner)
    refined_up, refined_down = zip(*refined_preds)
    refined_up = np.array(refined_up)
    refined_down = np.array(refined_down)

    recent_df = df.tail(len(refined_up))  # match with edge length
    timestamps = recent_df.index[-len(refined_up):]  # refined predictions align with last N candles

    # =========== Return Outputs ============
    open = df["Open"].iloc[-len(refined_up):].values
    high = df["High"].iloc[-len(refined_up):].values
    low = df["Low"].iloc[-len(refined_up):].values
    close = df["Close"].iloc[-len(refined_up):].values
    volume = df["Volume"].iloc[-len(refined_up):].values
    recent_prices = np.column_stack((open, high, low, close, volume))

    return refined_up, refined_down, timestamps, recent_prices

if __name__ == "__main__":
    main("2025-03-01")