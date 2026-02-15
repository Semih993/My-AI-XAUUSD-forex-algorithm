import os
import csv
from datetime import datetime
import requests

# === TELEGRAM SETTINGS ===
# Enter own credentials

TELEGRAM_TOKEN = "" 
TELEGRAM_CHAT_ID = ""

def send_telegram(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message}
    try:
        requests.post(url, data=payload, timeout=5)
    except Exception as e:
        print(f"[Telegram Error] {e}")

def live_signal(flips_with_direction, prices, times, momentum, minimum_confidence, account_balance=103000):

    trade_file = "open_trades.csv"
    processed_flips_file = "processed_flips.csv"

    TP = 0.006
    SL = 0.006
    max_allowed_trades = 2

    # Load processed flips to avoid re-processing
    processed_flip_indices = set()
    if os.path.exists(processed_flips_file):
        with open(processed_flips_file, mode='r') as f:
            reader = csv.reader(f)
            for row in reader:
                if row:
                    processed_flip_indices.add(int(row[0]))

    # Load open trades
    open_trades = []
    if os.path.exists(trade_file):
        with open(trade_file, mode='r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                open_trades.append({
                    "entry_index": int(row["entry_index"]),
                    "entry_time": datetime.strptime(row["entry_time"], "%Y-%m-%d %H:%M:%S"),
                    "entry_price": float(row["entry_price"]),
                    "position_size": float(row["position_size"]),
                    "account_balance": float(row["account_balance"]),
                    "holding": row["holding"] == "True",
                    "max_price": float(row["max_price"]) if row["max_price"] != "None" else None
                })

    idx = len(times) - 1  # Current candle index
    now = times[idx]
    high = prices[idx][1]
    low = prices[idx][2]
    close = prices[idx][3]

    updated_trades = []

    for trade in open_trades:
        entry_index = trade["entry_index"]
        entry_time = trade["entry_time"]
        entry_price = trade["entry_price"]
        position_size = trade["position_size"]
        entry_balance = trade["account_balance"]
        holding = trade["holding"]
        max_price = trade["max_price"]

        if holding:
            max_price = max(max_price, high)
            drawdown = (max_price - high) / max_price
            if drawdown >= 0.002:
                gain = position_size * (max_price - entry_price) / entry_price
                final_balance = entry_balance + gain
                gain_pct = (max_price - entry_price) / entry_price * 100
                send_telegram(f"🚨 BULL RUN EXIT (TRAILING)\nEntry: {entry_time.strftime('%Y-%m-%d %H:%M:%S')} at {entry_price:.2f}\nExit: {now.strftime('%Y-%m-%d %H:%M:%S')} at {high:.2f}\nPnL: ${gain:,.2f} ({gain_pct:.2f}%)")
                continue
            else:
                updated_trades.append({**trade, "max_price": max_price})
                continue

        if high >= entry_price * (1 + TP):
            up_bodies = []
            down_bodies = []
            for k in range(max(idx - 4, 0), idx + 1):
                body = abs(prices[k][0] - prices[k][3])
                if prices[k][3] > prices[k][0]:
                    up_bodies.append(body)
                elif prices[k][3] < prices[k][0]:
                    down_bodies.append(body)
            sum_up = sum(up_bodies)
            sum_down = sum(down_bodies) if down_bodies else 1e-6

            if sum_up / sum_down >= 2.5:
                updated_trades.append({**trade, "holding": True, "max_price": high})
                send_telegram(f"🔥 BULL RUN DETECTED\nHolding after TP\nEntry: {entry_time.strftime('%Y-%m-%d %H:%M:%S')} at {entry_price:.2f}\nHigh: {high:.2f}")
                continue
            else:
                gain = position_size * TP
                final_balance = entry_balance + gain
                gain_pct = TP * 100
                send_telegram(f"✅ TAKE PROFIT\nEntry: {entry_time.strftime('%Y-%m-%d %H:%M:%S')} at {entry_price:.2f}\nExit: {now.strftime('%Y-%m-%d %H:%M:%S')} at {entry_price*(1+TP):.2f}\nPnL: ${gain:,.2f} ({gain_pct:.2f}%)")
                continue

        elif low <= entry_price * (1 - SL):
            loss = -position_size * SL
            final_balance = entry_balance + loss
            loss_pct = SL * 100
            send_telegram(f"🛑 STOP LOSS\nEntry: {entry_time.strftime('%Y-%m-%d %H:%M:%S')} at {entry_price:.2f}\nExit: {now.strftime('%Y-%m-%d %H:%M:%S')} at {entry_price*(1-SL):.2f}\nPnL: ${loss:,.2f} (-{loss_pct:.2f}%)")
            continue

        updated_trades.append(trade)

    # Save open trades
    with open(trade_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["entry_index", "entry_time", "entry_price", "position_size", "account_balance", "holding", "max_price"])
        for t in updated_trades:
            writer.writerow([
                t["entry_index"],
                t["entry_time"].strftime("%Y-%m-%d %H:%M:%S"),
                round(t["entry_price"], 5),
                round(t["position_size"], 2),
                round(t["account_balance"], 2),
                str(t["holding"]),
                round(t["max_price"], 5) if t["max_price"] is not None else "None"
            ])

    # Entry logic for current candle only
    if len(updated_trades) < max_allowed_trades:
        matching_flips = [f for f in flips_with_direction if f[0] == idx and f[0] not in processed_flip_indices]

        if not matching_flips:
            send_telegram(f"No trade signal at {times[idx].strftime('%Y-%m-%d %H:%M:%S')}")
            return

        for flip in matching_flips:
            _, _, no_trade_pred, trade_pred = flip
            confidence = max(no_trade_pred, trade_pred) / min(no_trade_pred, trade_pred)
            signal_strength = momentum[idx] - momentum[idx - 2] if idx >= 2 else 0

            should_trade = (
                abs(trade_pred) > abs(no_trade_pred)
                and confidence >= minimum_confidence
                and signal_strength <= -0.005
            )

            if should_trade:
                winrate = 0.65
                reward = TP
                risk = SL
                kelly_fraction = winrate - (1 - winrate) * (reward / risk)
                kelly_fraction = min(kelly_fraction, 1.0)
                position_size = account_balance * kelly_fraction * 2.0
                entry_price = prices[idx][0]

                send_telegram(f"📈 TRADE ENTERED\nTime: {times[idx].strftime('%Y-%m-%d %H:%M:%S')}\nPrice: {entry_price:.2f}\nSize: ${position_size:,.0f}\nConfidence: {confidence:.2f}")

                updated_trades.append({
                    "entry_index": idx,
                    "entry_time": times[idx],
                    "entry_price": entry_price,
                    "position_size": position_size,
                    "account_balance": account_balance,
                    "holding": False,
                    "max_price": None
                })

                with open(trade_file, mode='w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(["entry_index", "entry_time", "entry_price", "position_size", "account_balance", "holding", "max_price"])
                    for t in updated_trades:
                        writer.writerow([
                            t["entry_index"],
                            t["entry_time"].strftime("%Y-%m-%d %H:%M:%S"),
                            round(t["entry_price"], 5),
                            round(t["position_size"], 2),
                            round(t["account_balance"], 2),
                            str(t["holding"]),
                            round(t["max_price"], 5) if t["max_price"] is not None else "None"
                        ])

                with open(processed_flips_file, mode='a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([idx])

