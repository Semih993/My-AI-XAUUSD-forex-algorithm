import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import time
from datetime import datetime, timedelta
from test_model import main
from scipy.signal import hilbert
from collections import defaultdict
from flip_direction import build_direction_dataset, train_model, FlipDirectionNet
import os
import torch
import torch.nn.functional as F
import mplfinance as mpf
import requests
from torch import nn
from torch.utils.data import Dataset, DataLoader
import requests
from sklearn.model_selection import train_test_split

# ========== Telegram Config ==========
TELEGRAM_TOKEN = "7605615062:AAEPIqm0ZmhUJ7o1S-GgPiK1WSRikgwmhKM"
TELEGRAM_CHAT_ID = "6404667335"

def send_telegram_message(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {'chat_id': TELEGRAM_CHAT_ID, 'text': message}
    try:
        requests.post(url, data=payload)
    except Exception as e:
        print(f"Telegram Error: {e}")

def get_entries(times, up, down, prices, extent=-4700):
    """
    Detects flips based on a change from positive to negative in the entry_exit signal.
    Prints the datetime, up/down values, and inverted direction of flips detected in the last 800 candles.
    """
    times_full = times  # Full dataset for index reference
    up_full = np.array(up)
    down_full = np.array(down)

    times = times[extent:]
    up = np.array(up[extent:])
    down = np.array(down[extent:])
    prices = prices[extent:]

    # === Entry/Exit Signal Calculation ===
    signal = up - down
    entry_exit = signal #pd.Series(signal).rolling(window=3).mean().fillna(0).values
    entry_exit = entry_exit * np.sign(np.gradient(entry_exit))

    flips = []

    def floor_to_half_hour(dt):
        return dt.replace(minute=0 if dt.minute < 30 else 30, second=0, microsecond=0)

    allowed_slots = {
        '00:00', '00:30', '01:00', '01:30', '02:00', '02:30', '03:00', '03:30',
        '04:00', '04:30', '05:00', '05:30', '06:00', '06:30', '07:00', '07:30',
        '08:00', '08:30', '09:00', '09:30', '10:00', '10:30', '11:00', '11:30',
        '12:00', '12:30', '13:00', '13:30', '14:00', '14:30', '15:00', '15:30',
        '16:00', '16:30', '17:00', '17:30', '18:00', '18:30', '19:00', '19:30',
        '20:00', '20:30', '21:00', '21:30', '22:00', '22:30', '23:00', '23:30'
    }

    for i in range(2, len(entry_exit)):
        if entry_exit[i - 1] > 0 and entry_exit[i] < 0:
            idx = i
            timestamp = times[idx]
            time_slot = floor_to_half_hour(timestamp).strftime("%H:%M")

            if time_slot not in allowed_slots:
                continue

            absolute_idx = len(times_full) - len(times) + idx
            flips.append(absolute_idx)

    # === Print Flips With Inverted Direction in the Last 800 Candles ===
    plot_candles = 800
    """
    print("\n📅 Flips Detected in the Last 800 Candles:")
    for idx in flips:
        if idx >= len(times_full) - plot_candles:
            flip_time = times_full[idx]

            print(f"- {flip_time}")
    """
    return flips

def generate_hilbert_signals(price_series):
    """
    Recreates the TradingView Hilbert Transformer Indicator using recursive filtering.
    """
    lpperiod = 20
    filttop = 48

    pi = np.pi
    _z = 0.707 * 2 * pi / filttop
    alpha1 = (np.cos(_z) + np.sin(_z) - 1) / np.cos(_z)

    HP = np.zeros_like(price_series)
    filt = np.zeros_like(price_series)

    for i in range(2, len(price_series)):
        HP[i] = (1 - alpha1 / 2) ** 2 * (price_series[i] - 2 * price_series[i - 1] + price_series[i - 2]) + \
                2 * (1 - alpha1) * HP[i - 1] - (1 - alpha1) ** 2 * HP[i - 2]

    a1 = np.exp(-1.414 * pi / lpperiod)
    b1 = 2 * a1 * np.cos(1.414 * pi / lpperiod)
    c2 = b1
    c3 = -a1 ** 2
    c1 = 1 - c2 - c3

    for i in range(2, len(HP)):
        filt[i] = c1 * (HP[i] + HP[i - 1]) / 2 + c2 * filt[i - 1] + c3 * filt[i - 2]

    ipeak = np.zeros_like(price_series)
    for i in range(1, len(filt)):
        ipeak[i] = max(abs(filt[i]), 0.991 * ipeak[i - 1])
    real = np.divide(filt, ipeak, out=np.zeros_like(filt), where=ipeak != 0)

    quadrature = np.zeros_like(price_series)
    for i in range(1, len(real)):
        quadrature[i] = real[i] - real[i - 1]

    qpeak = np.zeros_like(price_series)
    for i in range(1, len(quadrature)):
        qpeak[i] = max(abs(quadrature[i]), 0.991 * qpeak[i - 1])
    quadrature = np.divide(quadrature, qpeak, out=np.zeros_like(quadrature), where=qpeak != 0)

    _a1 = np.exp(-1.414 * pi / 10)
    _b1 = 2 * _a1 * np.cos(1.414 * pi / 10)
    _c2 = _b1
    _c3 = -_a1 ** 2
    _c1 = 1 - _c2 - _c3

    imag = np.zeros_like(price_series)
    for i in range(2, len(quadrature)):
        imag[i] = _c1 * (quadrature[i] + quadrature[i - 1]) / 2 + _c2 * imag[i - 1] + _c3 * imag[i - 2]

    return real, imag

def calculate_indicators(prices, window=14):
    # Convert to DataFrame for easier calculations
    df = pd.DataFrame(prices, columns=['open', 'high', 'low', 'close', 'volume'])
    
    # Calculate ATR
    df['high_low'] = df['high'] - df['low']
    df['high_close'] = np.abs(df['high'] - df['close'].shift())
    df['low_close'] = np.abs(df['low'] - df['close'].shift())
    df['tr'] = df[['high_low', 'high_close', 'low_close']].max(axis=1)
    df['ATR'] = df['tr'].rolling(window=window).mean().fillna(0)
    
    # Calculate Directional Movement
    df['up_move'] = df['high'] - df['high'].shift()
    df['down_move'] = df['low'].shift() - df['low']

    df['plus_dm'] = np.where((df['up_move'] > df['down_move']) & (df['up_move'] > 0), df['up_move'], 0)
    df['minus_dm'] = np.where((df['down_move'] > df['up_move']) & (df['down_move'] > 0), df['down_move'], 0)

    df['plus_di'] = 100 * (df['plus_dm'].rolling(window=window).sum() / df['ATR'])
    df['minus_di'] = 100 * (df['minus_dm'].rolling(window=window).sum() / df['ATR'])

    df['plus_di'] = df['plus_di'].fillna(0)
    df['minus_di'] = df['minus_di'].fillna(0)

    return df['ATR'].values, df['plus_di'].values, df['minus_di'].values

# ================= Live Direction Prediction =================
class DirectionDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        X, y = self.samples[idx]
        return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long)
    
class Attention(nn.Module):
    def __init__(self, input_dim):
        super(Attention, self).__init__()
        self.W = nn.Linear(input_dim, 256)
        self.V = nn.Linear(256, 1)

    def forward(self, x):
        # x: [batch_size, channels, sequence_length]
        x = x.permute(0, 2, 1)  # [batch_size, sequence_length, channels]

        energy = torch.tanh(self.W(x))  # [batch_size, sequence_length, 128]
        attention_scores = self.V(energy).squeeze(-1)  # [batch_size, sequence_length]

        attention_weights = torch.softmax(attention_scores, dim=1)  # [batch_size, sequence_length]

        attention_weights = attention_weights.unsqueeze(-1)  # [batch_size, sequence_length, 1]
        context_vector = (x * attention_weights).sum(dim=1)  # [batch_size, channels]

        return context_vector
    
class FlipDirectionNet(nn.Module):
    def __init__(self, input_features=10):
        super(FlipDirectionNet, self).__init__()
        self.conv1 = nn.Conv1d(input_features, 256, kernel_size=9, padding=4)
        self.conv2 = nn.Conv1d(256, 400, kernel_size=9, padding=4)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(400, 2)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # (batch_size, features, sequence_length)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.global_pool(x).squeeze(-1)
        x = self.dropout(x)
        x = self.fc(x)
        return x
    
def prepare_features(prices, real, imag, atr, plus_di, minus_di, idx, window=30):
    if idx - window < 0:
        return None

    sequence_opens = [prices[i][0] for i in range(idx - window, idx)]
    sequence_highs = [prices[i][1] for i in range(idx - window, idx)]
    sequence_lows = [prices[i][2] for i in range(idx - window, idx)]
    sequence_closes = [prices[i][3] for i in range(idx - window, idx)]
    sequence_volumes = [prices[i][4] for i in range(idx - window, idx)]
    sequence_real = real[idx - window:idx]
    sequence_imag = imag[idx - window:idx]
    sequence_atr = atr[idx - window:idx]
    sequence_plus_di = plus_di[idx - window:idx]
    sequence_minus_di = minus_di[idx - window:idx]

    X = torch.tensor(list(zip(
        sequence_opens,
        sequence_highs,
        sequence_lows,
        sequence_closes,
        sequence_volumes,
        sequence_real,
        sequence_imag,
        sequence_atr,
        sequence_plus_di,
        sequence_minus_di
    )), dtype=torch.float32)

    mean = X.mean(dim=0, keepdim=True)
    std = X.std(dim=0, keepdim=True) + 1e-8
    X = (X - mean) / std

    return X.numpy()

# ================= Training Function =================

def train_model(model, device, flips, prices, real, imag, atr, plus_di, minus_di):
    samples = []
    flip_indices = []  # Track flip indices for validation backtest

    for idx in flips:
        if idx - 15 < 0:
            continue

        entry_price = prices[idx][0]
        tp_level = entry_price * (1 + 0.0030)
        sl_level = entry_price * (1 - 0.0020)

        label = None

        for future_idx in range(idx + 1, len(prices)):
            high = prices[future_idx][1]
            low = prices[future_idx][2]

            if high >= tp_level:
                label = 1
                break
            if low <= sl_level:
                label = 0
                break

        if label is None:
            continue

        X = prepare_features(prices, real, imag, atr, plus_di, minus_di, idx)
        if X is None:
            continue  # Skip invalid samples
        samples.append((X, label))
        flip_indices.append(idx)

    if len(samples) == 0:
        print("No valid samples for training.")
        return None, None, None

    # === Time-Based Split ===
    split_index = int(len(samples) * 0.8)
    train_samples = samples[:split_index]
    val_samples = samples[split_index:]
    train_flip_indices = flip_indices[:split_index]
    val_flip_indices = flip_indices[split_index:]

    train_loader = DataLoader(DirectionDataset(train_samples), batch_size=32, shuffle=True)
    val_loader = DataLoader(DirectionDataset(val_samples), batch_size=32, shuffle=False)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

    epochs = 80
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            predicted = torch.argmax(outputs, dim=1)
            correct += (predicted == y_batch).sum().item()
            total += y_batch.size(0)

        train_accuracy = 100 * correct / total

        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item()

                predicted = torch.argmax(outputs, dim=1)
                val_correct += (predicted == y_batch).sum().item()
                val_total += y_batch.size(0)

        val_accuracy = 100 * val_correct / val_total

        print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {total_loss / len(train_loader):.6f}, Train Acc: {train_accuracy:.2f}%, Val Loss: {val_loss / len(val_loader):.6f}, Val Acc: {val_accuracy:.2f}%")

    torch.save(model.state_dict(), 'flip_direction_model.pth')
    print("Model training completed and saved.")

    return model, val_samples, val_flip_indices

def prepare_live_input(prices, real, imag, atr, plus_di, minus_di, idx):
    X = prepare_features(prices, real, imag, atr, plus_di, minus_di, idx)
    if X is None:
        return None
    X = torch.tensor(X, dtype=torch.float32).unsqueeze(0)
    return X

# ================= Test Function =================

def test_model(model, device, test_samples, min_confidence=0.6):
    if len(test_samples) == 0:
        print("No test samples available.")
        return

    test_loader = DataLoader(DirectionDataset(test_samples), batch_size=32, shuffle=False)

    model.eval()
    total = 0
    correct = 0
    skipped = 0

    print("\n=== Test Predictions ===")
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            probabilities = torch.softmax(outputs, dim=1)
            confidences, predicted = torch.max(probabilities, dim=1)

            for i in range(len(predicted)):
                if confidences[i].item() < min_confidence:
                    skipped += 1
                    continue

                total += 1
                if predicted[i].item() == y_batch[i].item():
                    correct += 1

                pred_dir = 'LONG' if predicted[i].item() == 1 else 'SHORT'
                true_dir = 'LONG' if y_batch[i].item() == 1 else 'SHORT'
                print(f"Predicted: {pred_dir} | Actual: {true_dir} | Confidence: {confidences[i].item():.2f}")

    if total > 0:
        test_accuracy = 100 * correct / total
        print(f"\nTest Accuracy (Confidence ≥ {min_confidence}): {test_accuracy:.2f}%")
    else:
        print("\nNo test samples passed the confidence filter.")

    print(f"Skipped samples due to low confidence: {skipped}")

# ================= Sleep Until Next Half Hour =================

def sleep_until_next_half_hour(buffer_sec=10):
    """
    Sleeps until 5 seconds after the next candle close time (00 or 30 min).
    """
    now = datetime.utcnow()
    minute = now.minute
    second = now.second

    if minute < 30:
        next_close = now.replace(minute=30, second=buffer_sec, microsecond=0)
    else:
        next_hour = now + timedelta(hours=1)
        next_close = next_hour.replace(minute=0, second=buffer_sec, microsecond=0)

    sleep_duration = (next_close - now).total_seconds()
    if sleep_duration > 0:
        print(f"Sleeping for {sleep_duration:.1f} seconds until {next_close.time()}...")
        time.sleep(sleep_duration)

# ================= Live Execution =================

def advanced_backtest(flips, times, prices, model, device, atr, plus_di, minus_di,
                      min_confidence=0.7):

    if len(flips) == 0:
        print("No flips provided for backtest.")
        return

    close_prices = [c[3] for c in prices]
    real, imag = generate_hilbert_signals(close_prices)

    total_trades = wins = losses = 0
    profit_pc = loose_pc = 0
    loose_streak = 0
    model_prediction_list = []

    for flip_idx in flips:
        if flip_idx + 1 >= len(prices) or flip_idx - 15 < 0:
            continue

        X = prepare_live_input(prices, real, imag, atr, plus_di, minus_di, flip_idx)
        if X is None:
            continue

        X = X.to(device)
        prediction = torch.softmax(model(X), dim=1)
        model_direction = torch.argmax(prediction, dim=1).item()
        confidence = prediction[0][model_direction].item()

        if confidence < min_confidence:
            continue
        model_prediction_list.append(model_direction)

        average_direction = prices[flip_idx][3] - prices[flip_idx-10][3]
        average_direction = 0 if average_direction < 0 else 1

        """
        if loose_streak == 2:
            if (model_direction != average_direction) and model_prediction_list[-1] != average_direction:
                model_direction = average_direction
            loose_streak = 0
        """
        
        if model_direction != average_direction:
            continue
        
        sl, tp = 0.0030, 0.0030
        
        final_direction = model_direction
    
        entry_time = times[flip_idx + 1]
        entry_price = prices[flip_idx + 1][0]

        if final_direction == 1:
            tp_level = entry_price * (1 + tp)
            sl_level = entry_price * (1 - sl)
        else:
            tp_level = entry_price * (1 - tp)
            sl_level = entry_price * (1 + sl)     

        result = None

        for future_idx in range(flip_idx + 1, len(prices)):
            high, low = prices[future_idx][1], prices[future_idx][2]

            if final_direction == 1:
                if low <= sl_level:
                    result = 'SL'
                    losses += 1
                    loose_pc += sl
                    loose_streak += 1
                    break
                if high >= tp_level:
                    result = 'TP'
                    wins += 1
                    profit_pc += tp
                    break
            else:
                if high >= sl_level:
                    result = 'SL'
                    loose_streak += 1
                    losses += 1
                    loose_pc += sl
                    break
                if low <= tp_level:
                    result = 'TP'
                    wins += 1
                    profit_pc += tp
                    break
            
        if result is None:
            final_close = prices[-1][3]
            if final_direction == 1 and final_close >= entry_price:
                result = 'TP'
                wins += 1
                profit_pc += tp
            elif final_direction == 0 and final_close <= entry_price:
                result = 'TP'
                wins += 1
                profit_pc += tp
            else:
                result = 'SL'
                losses += 1
                loose_pc += sl
                loose_streak += 1

        total_trades += 1
        direction_str = 'Long' if final_direction == 1 else 'Short'
        emoji = "✅" if result == 'TP' else "❌"
        print(f"{emoji} {entry_time} | Direction: {direction_str:5} | Result: {result} | SL: {sl*100} TP: {tp*100}")

    success_rate = 100 * wins / total_trades if total_trades > 0 else 0
    print("\n====== Galactic Backtest Summary ======")
    print(f"Total Trades Executed: {total_trades}")
    print(f"Wins: {wins}")
    print(f"Losses: {losses}")
    print(f"Success Rate: {success_rate:.2f}%")
    print(f"Profits: {round(100 * profit_pc, 2)}%")
    print(f"Losses: {round(100 * loose_pc, 2)}%")
    print(f"Net Profit: {round(100 * (profit_pc - loose_pc), 2)}%")

    return success_rate

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FlipDirectionNet(input_features=10).to(device)
TRAIN = True

if TRAIN or not os.path.exists('flip_direction_model.pth'):
    print("Model file not found or training requested. Starting training...")
    up, down, times, prices = main(date='2025-03-01')
    ATR_values, plus_di, minus_di = calculate_indicators(prices)
    flips = get_entries(times, up, down, prices)
    close_prices = [c[3] for c in prices]
    real, imag = generate_hilbert_signals(close_prices)

    model, test_samples, val_flip_indices = train_model(model, device, flips, prices, real, imag, ATR_values, plus_di, minus_di)
    test_model(model, device, test_samples, min_confidence=0.75)

model.load_state_dict(torch.load('flip_direction_model.pth'))
model.eval()

def write_trade_signal(direction, lot_size):
    with open("C:/Users/arsla/AppData/Roaming/MetaQuotes/Terminal/Common/Files/trade_signal.txt", "w") as f:
        print(os.path.expanduser("~"))
        f.write(f"{direction},{lot_size}")
#write_trade_signal("BUY", 0.01)

processed_flips = set()

while True:
    try:
        up, down, times, prices = main(date="2025-05-01")

        if len(times) >= 100:
            ATR_values, plus_di, minus_di = calculate_indicators(prices)
            flips = get_entries(times, up, down, prices)

            advanced_backtest(flips, times, prices, model, device, ATR_values, plus_di, minus_di,
                                min_confidence=0.7)

            close_prices = [c[3] for c in prices]
            real, imag = generate_hilbert_signals(close_prices)

            latest_index = len(times) - 1
            latest_time = times[latest_index] + timedelta(minutes=30)
            execution_price = prices[latest_index][3]

            entry_sent = False

            for flip_idx in flips:
                # Only act on the candle that just closed
                # print(f"index: {flip_idx}, len: {len(times)}")
                if (flip_idx < len(times) - 2):
                    continue

                processed_flips.add(flip_idx)

                X = prepare_live_input(prices, real, imag, ATR_values, plus_di, minus_di, flip_idx)
                if X is None:
                    continue

                X = X.to(device)
                prediction = torch.softmax(model(X), dim=1)
                model_direction = torch.argmax(prediction, dim=1).item()
                confidence = prediction[0][model_direction].item()

                if confidence < 0.7:
                    continue

                dir_str = 'BUY' if model_direction == 1 else 'SELL'
                entry_price = prices[flip_idx][3]
                entry_time = times[flip_idx]

                message = (
                    f"✅ Entry Detected\n\n"
                    f"🕒 Entry Time: {entry_time}\n"
                    f"💰 Entry Price: {entry_price:.2f}\n"
                    f"📊 Direction: {dir_str}"
                )
                print(message)
                send_telegram_message(message)
                entry_sent = True

                break

            if not entry_sent:
                latest_time = times[-1]
                execution_price = prices[-1][3]

                message = (
                    f"❌ No Trades Detected!\n\n"
                    f"🕒 Checked Time: {latest_time}\n"
                    f"💰 Latest Price: {execution_price:.2f}"
                )
                print(message)
                send_telegram_message(message)

    except Exception as e:
        print(f"[{datetime.utcnow()}] Error occurred: {e}")
        send_telegram_message(f"⚠️ Error occurred: {e}")

    sleep_until_next_half_hour()