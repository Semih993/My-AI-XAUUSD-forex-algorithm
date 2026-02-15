import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from collections import defaultdict
import datetime

def floor_to_half_hour(dt):
    return dt.replace(minute=0 if dt.minute < 30 else 30, second=0, microsecond=0)

def simulate_profit_smart(flips, prices, timestamps, momentum):

    flips = [f for f in flips if f[1] != 0]
    print(f"NO. OF ENTRIES (initial): {len(flips)}")

    candles = -4700
    prices = prices[candles:]
    timestamps = timestamps[candles:]

    start_date = timestamps[0].date()
    end_date = timestamps[-1].date()
    num_days = (end_date - start_date).days or 1
    print(f"Simulation period: {start_date} to {end_date} ({num_days} days)")

    open_prices = prices[:, 0]
    high_prices = prices[:, 1]
    low_prices = prices[:, 2]

    allowed_slots = {
        '00:00', '00:30', '01:00', '01:30', '02:00', '02:30', '03:00', '03:30', 
        '04:00', '04:30', '05:00', '05:30', '06:00', '06:30', '07:00', '07:30', 
        '08:00', '08:30', '09:00', '09:30', '10:00', '10:30', '11:00', '11:30', 
        '12:00', '12:30', '13:00', '13:30', '14:00', '14:30', '15:00', '15:30',
        '16:00', '16:30', '17:00', '17:30', '18:00', '18:30', '19:00', '19:30',
        '20:00', '20:30', '21:00', '22:00', '22:30', '23:00', '23:30'
    }

    time_bins = defaultdict(lambda: {"success": 0, "fail": 0, 'effective_success': 0.0})
    all_slots = [f"{str(h).zfill(2)}:{m}" for h in range(24) for m in ("00", "30")]

    # ------------- GET CONFIDENCE LEVELS ------------
    
    confidence = [0] * len(flips)
    for i, (index, direction, no_trade_pred, trade_pred) in enumerate(flips):
        confidence[i] = max(no_trade_pred, trade_pred)/min(no_trade_pred, trade_pred)

    average = np.mean(confidence)
    std = np.std(confidence)
    minimum_confidence = average - (std * 1) 

    # ----------- MOMENTUM CONFIDENCE ------------

    momentum_delta = [0]*len(momentum)
    for i in range(len(momentum)):
        if i >= 2:
            momentum_delta[i] = momentum[i] - momentum[i-2]
        else:
            momentum_delta[i] = 0

    long = 0
    short = 0
    long_success = 0
    long_fail = 0
    short_success = 0
    short_fail = 0
    no_trade = 0
    approved_long_entries = []
    MOMENTUM_THRESHOLD = 0.0025
    EXIT_THRESHOLD = 0.0060

    # =============================== MAIN LOOP ==============================

    for i, (index, direction, no_trade_pred, trade_pred) in enumerate(flips):
        
        # ------------ CHECK TO QUALIFY ------------- 

        if index >= len(open_prices):
            continue
        entry_time = timestamps[index]
        time_slot = floor_to_half_hour(entry_time).strftime("%H:%M")
        if time_slot not in allowed_slots:
            continue

        # ------------- DIRECTION ASSIGNMENT --------------- 

        confidence = max(no_trade_pred, trade_pred)/min(no_trade_pred, trade_pred)
        entry_date = timestamps[index].date()
        entry_time = timestamps[index].time()

        if (abs(trade_pred) > abs(no_trade_pred)) and (confidence >= minimum_confidence) and (momentum_delta[index] <= -MOMENTUM_THRESHOLD):
            direction = "TRADE"
            long += 1
            approved_long_entries.append(index)
        else:
            direction = "NO-TRADE"
            no_trade += 1
            continue

        next_index = flips[i + 1][0] if i + 1 < len(flips) else len(prices) - 1
        entry_price = open_prices[index]
        
        j = 0
        success_flag = 0

        # --------- SESSION ITERATION --------

        while success_flag == 0 and (j + index) < len(open_prices):

            high = high_prices[index + j]
            low = low_prices[index + j]

            if direction == "TRADE":  
                if ((low - entry_price) / entry_price) < -EXIT_THRESHOLD:
                    time_bins[time_slot]["fail"] += 1
                    success_flag = -1
                    break
                elif ((high - entry_price) / entry_price) >= EXIT_THRESHOLD:
                    gain = (high - entry_price) / entry_price * 100
                    time_bins[time_slot]["success"] += 1
                    success_flag = 1
                    break
            j += 1

        if success_flag != 0:
            exit_date = timestamps[index + j].date()
            exit_time = timestamps[index + j].time()
            result = "WIN" if success_flag == 1 else "LOSS"

            if result == "WIN":
                long_success += 1
            else:
                long_fail += 1

            print(f"{entry_date} @ {entry_time} ---> {exit_date} @ {exit_time} ___ {result}")

    # === Print Result Table ===
    print(f"{'Time':>5} | {'Successes':>9} | {'Failures':>8} | {'Success Rate':>13}")
    print("-" * 70)
    total_success = total_fail = total_effective_success = 0.0

    for slot in all_slots:
        success = time_bins[slot]["success"]
        fail = time_bins[slot]["fail"]
        total = success + fail
        rate = (success / total) * 100 if total > 0 else None
        total_success += success
        total_fail += fail
        rate_str = f"{rate:.2f}%" if rate is not None else "  N/A"
        print(f"{slot:>5} | {success:>9} | {fail:>8} | {rate_str:>12}")


    
    print("\n" + "-" * 70)
    print(f"Simulation period: {start_date} to {end_date} ({num_days} days)")
    print(f"Trades per day: {total / num_days:.2f}")
    print(f"Long Trades: {long_success + long_fail}, Long Success: {long_success}, Long Fail: {long_fail}")
    print("=============================")
    print(f"OVERALL LONG WIN RATE: {round(100 * long_success/(long_success + long_fail + 0.001), 2)}%")
    print("=============================")
    print(f"Long: {long},  No-trade: {no_trade},  total: {long + no_trade}")




def floor_to_half_hour(dt):
    return dt.replace(minute=0 if dt.minute < 30 else 30, second=0, microsecond=0)

def simulate_account_growth(flips, prices, timestamps, momentum):
    flips = [f for f in flips if f[1] != 0]
    print(f"NO. OF ENTRIES (initial): {len(flips)}")

    candles = -4700
    prices = prices[candles:]
    timestamps = timestamps[candles:]

    start_date = timestamps[0].date()
    end_date = timestamps[-1].date()
    num_days = (end_date - start_date).days or 1

    open_prices = prices[:, 0]
    high_prices = prices[:, 1]
    low_prices = prices[:, 2]

    allowed_slots = {f"{str(h).zfill(2)}:{m}" for h in range(24) for m in ("00", "30")}

    confidence = [max(f[2], f[3]) / min(f[2], f[3]) for f in flips]
    avg_conf = np.mean(confidence)
    std_conf = np.std(confidence)
    min_conf = avg_conf - 0.3 * std_conf

    momentum_delta = [momentum[i] - momentum[i-2] if i >= 2 else 0 for i in range(len(momentum))]

    EXIT_THRESHOLD = 0.006
    account_balance = 100000
    trade_returns = []
    win_returns = []
    loss_returns = []
    trade_count = 0

    for i, (index, _, no_trade_pred, trade_pred) in enumerate(flips):
        if index >= len(open_prices):
            continue

        entry_time = timestamps[index]
        time_slot = floor_to_half_hour(entry_time).strftime("%H:%M")
        if time_slot not in allowed_slots:
            continue

        conf = max(no_trade_pred, trade_pred) / min(no_trade_pred, trade_pred)
        if (abs(trade_pred) <= abs(no_trade_pred)) or (conf < min_conf) or (momentum_delta[index] > -0.005):
            continue


        winrate = 0.65
        reward = 0.006
        risk = 0.006
        kelly_fraction = winrate - (1 - winrate) * (reward / risk)
        kelly_fraction = min(kelly_fraction, 1.0)
        position_size = account_balance * kelly_fraction * 1.5
        position_ratio = round(position_size / 1000, 0)
        print(f"   - Position Size: {position_ratio}%       Account Balance: £{round(account_balance , 0)}")

        entry_price = open_prices[index]
        j = 0

        while index + j < len(high_prices):
            high = high_prices[index + j]
            low = low_prices[index + j]

            if ((low - entry_price) / entry_price <= -EXIT_THRESHOLD):
                loss = -position_size * risk
                account_balance += loss
                trade_returns.append(loss)
                loss_returns.append(-loss)
                trade_count += 1
                break

            if ((high - entry_price) / entry_price >= EXIT_THRESHOLD):
                gain = position_size * reward
                account_balance += gain
                trade_returns.append(gain)
                win_returns.append(gain)
                trade_count += 1
                break

            j += 1

    profit = account_balance - 100000
    profit_factor = sum(win_returns) / sum(loss_returns) if loss_returns else float('inf')
    sharpe_ratio = np.mean(trade_returns) / (np.std(trade_returns) + 1e-6) * np.sqrt(len(trade_returns)) if len(trade_returns) > 1 else 0

    print("\n" + "=" * 60)
    print(f"Start Balance: $100,000")
    print(f"Final Balance: ${account_balance:,.2f}")
    print(f"Net Profit: ${profit:,.2f}")
    print(f"Number of Trades: {trade_count}")
    print(f"Simulation Period: {start_date} to {end_date} ({num_days} days)")
    print(f"Profit Factor: {profit_factor:.2f}")
    print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
    print("=" * 60)





def simulate_account_growth_refined(flips, prices, timestamps, momentum):
    import numpy as np
    import matplotlib.pyplot as plt

    MAXIMUM_SIMULTANEOUS_TRADES = 2

    flips = [f for f in flips if f[1] != 0]
    print(f"NO. OF ENTRIES (initial): {len(flips)}")

    candles = -4700
    prices = prices[candles:]
    timestamps = timestamps[candles:]

    start_date = timestamps[0].date()
    end_date = timestamps[-1].date()
    num_days = (end_date - start_date).days or 1

    open_prices = prices[:, 0]
    high_prices = prices[:, 1]
    low_prices = prices[:, 2]
    close_prices = prices[:, 3]

    allowed_slots = {f"{str(h).zfill(2)}:{m}" for h in range(24) for m in ("00", "30")}

    confidence = [max(f[2], f[3]) / min(f[2], f[3]) for f in flips]
    avg_conf = np.mean(confidence)
    std_conf = np.std(confidence)
    min_conf = avg_conf - 0.3 * std_conf

    momentum_delta = [momentum[i] - momentum[i-2] if i >= 2 else 0 for i in range(len(momentum))]

    EXIT_THRESHOLD = 0.006
    account_balance = 100000
    trade_returns = []
    win_returns = []
    loss_returns = []
    trade_count = 0

    equity_curve = [account_balance] * len(close_prices)
    open_trades = []

    for i in range(len(close_prices)):
        floating_pnl = 0
        high = high_prices[i]
        low = low_prices[i]

        updated_trades = []
        for trade in open_trades:
            index, entry_price, position_size, holding, max_price = trade
            price_change_up = (high - entry_price) / entry_price
            price_change_down = (low - entry_price) / entry_price

            if not holding:
                if price_change_down <= -EXIT_THRESHOLD:
                    loss = -position_size * EXIT_THRESHOLD
                    account_balance += loss
                    trade_returns.append(loss)
                    loss_returns.append(-loss)
                    trade_count += 1
                elif price_change_up >= EXIT_THRESHOLD:
                    lookback_range = range(max(i - 4, 0), i + 1)
                    up_bodies = []
                    down_bodies = []
                    for k in lookback_range:
                        body = abs(open_prices[k] - close_prices[k])
                        if close_prices[k] > open_prices[k]:
                            up_bodies.append(body)
                        elif close_prices[k] < open_prices[k]:
                            down_bodies.append(body)
                    sum_up = sum(up_bodies)
                    sum_down = sum(down_bodies) if down_bodies else 1e-6
                    if sum_up / sum_down >= 2.5:
                        updated_trades.append((index, entry_price, position_size, True, None))
                    else:
                        gain = position_size * EXIT_THRESHOLD
                        account_balance += gain
                        trade_returns.append(gain)
                        win_returns.append(gain)
                        trade_count += 1
                else:
                    updated_trades.append(trade)
            else:
                current_profit = (high - entry_price) / entry_price
                current_loss = (low - entry_price) / entry_price
                if current_profit < 0.01:
                    if current_profit >= 0.006 and low < entry_price:
                        trade_returns.append(0)
                        trade_count += 1
                    elif current_loss <= -0.002:
                        loss = -position_size * abs(current_loss)
                        account_balance += loss
                        trade_returns.append(loss)
                        loss_returns.append(-loss)
                        trade_count += 1
                    else:
                        updated_trades.append(trade)
                else:
                    max_price = max(max_price or high, high)
                    if (max_price - high) / max_price >= 0.002:
                        final_gain = (max_price - entry_price) / entry_price
                        gain = position_size * final_gain
                        account_balance += gain
                        trade_returns.append(gain)
                        win_returns.append(gain)
                        trade_count += 1
                    else:
                        updated_trades.append((index, entry_price, position_size, True, max_price))
                        floating_pnl += (close_prices[i] - entry_price) * position_size / entry_price
        open_trades = updated_trades
        equity_curve[i] = account_balance + floating_pnl

        if len(open_trades) < MAXIMUM_SIMULTANEOUS_TRADES:
            matching_flips = [f for f in flips if f[0] == i]
            for flip in matching_flips:
                _, _, no_trade_pred, trade_pred = flip
                conf = max(no_trade_pred, trade_pred) / min(no_trade_pred, trade_pred)
                if (abs(trade_pred) <= abs(no_trade_pred)) or (conf < min_conf) or (momentum_delta[i] > -0.005):
                    continue

                winrate = 0.65
                reward = 0.006
                risk = 0.006
                kelly_fraction = winrate - (1 - winrate) * (reward / risk)
                kelly_fraction = min(kelly_fraction, 1.0)

                if i < 24 * 15 * 2:
                    kelly_fraction *= 0.3

                position_size = account_balance * kelly_fraction * 1.75

                #print(f"   - Position Size: {round(position_size / 1000, 0)}%       Account Balance: £{round(account_balance , 0)}")

                print(f"[{timestamps[i]}] Entry signal confirmed — Entry Price: {open_prices[i]:.2f}, Position Size: £{position_size:,.2f}")
                open_trades.append((i, open_prices[i], position_size, False, None))
                if len(open_trades) >= MAXIMUM_SIMULTANEOUS_TRADES:
                    break

    profit = account_balance - 100000
    profit_factor = sum(win_returns) / sum(loss_returns) if loss_returns else float('inf')
    sharpe_ratio = np.mean(trade_returns) / (np.std(trade_returns) + 1e-6) * np.sqrt(len(trade_returns)) if len(trade_returns) > 1 else 0
    max_drawdown = max(100000 - np.min(equity_curve), 0)

    print("\n" + "=" * 60)
    print(f"Start Balance: $100,000")
    print(f"Final Balance: ${account_balance:,.2f}")
    print(f"Net Profit: ${profit:,.2f}")
    print(f"Number of Trades: {trade_count}")
    print(f"Simulation Period: {start_date} to {end_date} ({num_days} days)")
    print(f"Profit Factor: {profit_factor:.2f}")
    print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
    print(f"Max Drawdown: ${max_drawdown:,.2f}")
    print("=" * 60)

    plt.figure(figsize=(14, 6))
    plt.plot(timestamps, equity_curve, label="Account Balance")
    plt.title("Account Balance Over Time")
    plt.xlabel("Time")
    plt.ylabel("Balance ($)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()




def simulate_account_growth_refined_X(flips, prices, timestamps, momentum):
    import numpy as np
    import matplotlib.pyplot as plt
    from tabulate import tabulate

    MAXIMUM_SIMULTANEOUS_TRADES = 2

    flips = [f for f in flips if f[1] != 0]
    print(f"NO. OF ENTRIES (initial): {len(flips)}")

    candles = -4700
    prices = prices[candles:]
    timestamps = timestamps[candles:]

    start_date = timestamps[0].date()
    end_date = timestamps[-1].date()
    num_days = (end_date - start_date).days or 1

    open_prices = prices[:, 0]
    high_prices = prices[:, 1]
    low_prices = prices[:, 2]
    close_prices = prices[:, 3]

    allowed_slots = {f"{str(h).zfill(2)}:{m}" for h in range(24) for m in ("00", "30")}

    confidence = [max(f[2], f[3]) / min(f[2], f[3]) for f in flips]
    avg_conf = np.mean(confidence)
    std_conf = np.std(confidence)
    min_conf = avg_conf - 0.3 * std_conf


    momentum_delta = [momentum[i] - momentum[i-2] if i >= 2 else 0 for i in range(len(momentum))]

    EXIT_THRESHOLD = 0.006
    account_balance = 100000
    trade_returns = []
    win_returns = []
    loss_returns = []
    trade_count = 0
    trade_logs = []

    equity_curve = [account_balance] * len(close_prices)
    open_trades = []

    for i in range(len(close_prices)):
        floating_pnl = 0
        high = high_prices[i]
        low = low_prices[i]

        updated_trades = []
        for trade in open_trades:
            index, entry_price, position_size, holding, max_price = trade
            price_change_up = (high - entry_price) / entry_price
            price_change_down = (low - entry_price) / entry_price

            if not holding:
                if price_change_down <= -EXIT_THRESHOLD:
                    loss = -position_size * EXIT_THRESHOLD
                    account_balance += loss
                    trade_returns.append(loss)
                    loss_returns.append(-loss)
                    trade_logs.append({
                        'entry_time': timestamps[index],
                        'exit_time': timestamps[i],
                        'entry_price': entry_price,
                        'exit_price': entry_price * (1 - EXIT_THRESHOLD),
                        'pnl': loss,
                        'pnl_pct': -EXIT_THRESHOLD * 100,
                        'result': 'SL',
                        'duration_min': (timestamps[i] - timestamps[index]).total_seconds() / 60
                    })
                    trade_count += 1
                elif price_change_up >= EXIT_THRESHOLD:
                    lookback_range = range(max(i - 4, 0), i + 1)
                    up_bodies = []
                    down_bodies = []
                    for k in lookback_range:
                        body = abs(open_prices[k] - close_prices[k])
                        if close_prices[k] > open_prices[k]:
                            up_bodies.append(body)
                        elif close_prices[k] < open_prices[k]:
                            down_bodies.append(body)
                    sum_up = sum(up_bodies)
                    sum_down = sum(down_bodies) if down_bodies else 1e-6
                    if sum_up / sum_down >= 2.5:
                        updated_trades.append((index, entry_price, position_size, True, None))
                    else:
                        gain = position_size * EXIT_THRESHOLD
                        account_balance += gain
                        trade_returns.append(gain)
                        win_returns.append(gain)
                        trade_logs.append({
                            'entry_time': timestamps[index],
                            'exit_time': timestamps[i],
                            'entry_price': entry_price,
                            'exit_price': entry_price * (1 + EXIT_THRESHOLD),
                            'pnl': gain,
                            'pnl_pct': EXIT_THRESHOLD * 100,
                            'result': 'TP',
                            'duration_min': (timestamps[i] - timestamps[index]).total_seconds() / 60
                        })
                        trade_count += 1
                else:
                    updated_trades.append(trade)
            else:
                current_profit = (high - entry_price) / entry_price
                current_loss = (low - entry_price) / entry_price
                if current_profit < 0.01:
                    if current_profit >= 0.006 and low < entry_price:
                        trade_returns.append(0)
                        trade_logs.append({
                            'entry_time': timestamps[index],
                            'exit_time': timestamps[i],
                            'entry_price': entry_price,
                            'exit_price': close_prices[i],
                            'pnl': 0,
                            'pnl_pct': 0,
                            'result': 'Neutral',
                            'duration_min': (timestamps[i] - timestamps[index]).total_seconds() / 60
                        })
                        trade_count += 1
                    elif current_loss <= -0.002:
                        loss = -position_size * abs(current_loss)
                        account_balance += loss
                        trade_returns.append(loss)
                        loss_returns.append(-loss)
                        trade_logs.append({
                            'entry_time': timestamps[index],
                            'exit_time': timestamps[i],
                            'entry_price': entry_price,
                            'exit_price': low,
                            'pnl': loss,
                            'pnl_pct': current_loss * 100,
                            'result': 'SL',
                            'duration_min': (timestamps[i] - timestamps[index]).total_seconds() / 60
                        })
                        trade_count += 1
                    else:
                        updated_trades.append(trade)
                else:
                    max_price = max(max_price or high, high)
                    if (max_price - high) / max_price >= 0.002:
                        final_gain = (max_price - entry_price) / entry_price
                        gain = position_size * final_gain
                        account_balance += gain
                        trade_returns.append(gain)
                        win_returns.append(gain)
                        trade_logs.append({
                            'entry_time': timestamps[index],
                            'exit_time': timestamps[i],
                            'entry_price': entry_price,
                            'exit_price': max_price,
                            'pnl': gain,
                            'pnl_pct': final_gain * 100,
                            'result': 'Hold Exit',
                            'duration_min': (timestamps[i] - timestamps[index]).total_seconds() / 60
                        })
                        trade_count += 1
                    else:
                        updated_trades.append((index, entry_price, position_size, True, max_price))
                        floating_pnl += (close_prices[i] - entry_price) * position_size / entry_price
        open_trades = updated_trades
        equity_curve[i] = account_balance + floating_pnl

        if len(open_trades) < MAXIMUM_SIMULTANEOUS_TRADES:
            matching_flips = [f for f in flips if f[0] == i]
            for flip in matching_flips:
                _, _, no_trade_pred, trade_pred = flip
                conf = max(no_trade_pred, trade_pred) / min(no_trade_pred, trade_pred)
                if (abs(trade_pred) <= abs(no_trade_pred)) or (conf < min_conf) or (momentum_delta[i] > -0.005):
                    continue

                winrate = 0.65
                reward = 0.006
                risk = 0.006
                kelly_fraction = winrate - (1 - winrate) * (reward / risk)
                kelly_fraction = min(kelly_fraction, 1.0)

                position_size = account_balance * kelly_fraction * 2.0
                open_trades.append((i, open_prices[i], position_size, False, None))
                if len(open_trades) >= MAXIMUM_SIMULTANEOUS_TRADES:
                    break

    profit = account_balance - 100000
    profit_factor = sum(win_returns) / sum(loss_returns) if loss_returns else float('inf')
    sharpe_ratio = np.mean(trade_returns) / (np.std(trade_returns) + 1e-6) * np.sqrt(len(trade_returns)) if len(trade_returns) > 1 else 0
    max_drawdown = max(100000 - np.min(equity_curve), 0)

    print("\n" + "=" * 60)
    print(f"Start Balance: $100,000")
    print(f"Final Balance: ${account_balance:,.2f}")
    print(f"Net Profit: ${profit:,.2f}")
    print(f"Number of Trades: {trade_count}")
    print(f"Simulation Period: {start_date} to {end_date} ({num_days} days)")
    print(f"Profit Factor: {profit_factor:.2f}")
    print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
    print(f"Max Drawdown: ${max_drawdown:,.2f}")
    print("=" * 60)
    """
    plt.figure(figsize=(14, 6))
    plt.plot(timestamps, equity_curve, label="Account Balance")
    plt.title("Account Balance Over Time")
    plt.xlabel("Time")
    plt.ylabel("Balance ($)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
    """
    # === Print all trade entry times ===
    print("\n--- ALL TRADE ENTRY TIMES ---")
    for i, log in enumerate(trade_logs, 1):
        entry_ts = log['entry_time']
        print(f"{i:>3}. {entry_ts.strftime('%Y-%m-%d %H:%M:%S')}")
    print("--------------------------------\n")
    

    return min_conf




"""
from collections import defaultdict
import numpy as np

def floor_to_half_hour(dt):
    return dt.replace(minute=0 if dt.minute < 30 else 30, second=0, microsecond=0)

def simulate_profit_smart(flips, prices, timestamps, momentum):

    flips = [f for f in flips if f[1] != 0]
    print(f"NO. OF ENTRIES (initial): {len(flips)}")

    candles = -4700
    prices = prices[candles:]
    timestamps = timestamps[candles:]

    start_date = timestamps[0].date()
    end_date = timestamps[-1].date()
    num_days = (end_date - start_date).days or 1
    print(f"Simulation period: {start_date} to {end_date} ({num_days} days)")

    open_prices = prices[:, 0]
    high_prices = prices[:, 1]
    low_prices = prices[:, 2]

    allowed_slots = {
        '00:00', '00:30', '01:00', '01:30', '02:00', '02:30', '03:00', '03:30', 
        '04:00', '04:30', '05:00', '05:30', '06:00', '06:30', '07:00', '07:30', 
        '08:00', '08:30', '09:00', '09:30', '10:00', '10:30', '11:00', '11:30', 
        '12:00', '12:30', '13:00', '13:30', '14:00', '14:30', '15:00', '15:30',
        '16:00', '16:30', '17:00', '17:30', '18:00', '18:30', '19:00', '19:30',
        '20:00', '20:30', '21:00', '22:00', '22:30', '23:00', '23:30'
    }

    time_bins = defaultdict(lambda: {"success": 0, "fail": 0, 'effective_success': 0.0})
    all_slots = [f"{str(h).zfill(2)}:{m}" for h in range(24) for m in ("00", "30")]

    # ------------- GET CONFIDENCE LEVELS ------------
    
    confidence = [0] * len(flips)
    for i, (index, direction, pred_up, pred_down) in enumerate(flips):
        confidence[i] = max(pred_up, pred_down)/min(pred_up, pred_down)

    average = np.mean(confidence)
    std = np.std(confidence)
    minimum_confidence = average - (std * 0.3) 

    # ----------- MOMENTUM CONFIDENCE ------------

    momentum_delta = [0]*len(momentum)
    for i in range(len(momentum)):
        if i >= 2:
            momentum_delta[i] = momentum[i] - momentum[i-2]
        else:
            momentum_delta[i] = 0

    long = 0
    short = 0
    long_success = 0
    long_fail = 0
    short_success = 0
    short_fail = 0
    no_trade = 0
    approved_long_entries = []
    MOMENTUM_THRESHOLD = 0.005
    EXIT_THRESHOLD = 0.0060

    # =============================== MAIN LOOP ==============================

    for i, (index, direction, pred_up, pred_down) in enumerate(flips):
        
        # ------------ CHECK TO QUALIFY ------------- 

        if index >= len(open_prices):
            continue
        entry_time = timestamps[index]
        time_slot = floor_to_half_hour(entry_time).strftime("%H:%M")
        if time_slot not in allowed_slots:
            continue

        # ------------- DIRECTION ASSIGNMENT --------------- 

        confidence = max(pred_up, pred_down)/min(pred_up, pred_down)
        entry_date = timestamps[index].date()
        entry_time = timestamps[index].time()

        if (abs(pred_down) > abs(pred_up)) and (confidence >= minimum_confidence) and (momentum_delta[index] <= -MOMENTUM_THRESHOLD):
            direction = "TRADE"
            long += 1
            approved_long_entries.append(index)
        else:
            direction = "NO-TRADE"
            no_trade += 1
            continue

        next_index = flips[i + 1][0] if i + 1 < len(flips) else len(prices) - 1
        entry_price = open_prices[index]
        
        j = 0
        success_flag = 0

        # --------- SESSION ITERATION --------

        while success_flag == 0:

            high = high_prices[index + j]
            low = low_prices[index + j]

            if direction == "TRADE":  
                if ((low - entry_price) / entry_price) < -EXIT_THRESHOLD:
                    time_bins[time_slot]["fail"] += 1
                    success_flag = -1
                    break
                elif ((high - entry_price) / entry_price) >= EXIT_THRESHOLD:
                    gain = (high - entry_price) / entry_price * 100
                    time_bins[time_slot]["success"] += 1
                    success_flag = 1
                    break
            j += 1

        exit_date = timestamps[index + j].date()
        exit_time = timestamps[index + j].time()

        if success_flag == 1:
            result = "WIN"
        elif success_flag == -1:
            result = "LOSS"

        if direction == "TRADE":
            if result == "WIN":
                long_success += 1
            elif result == "LOSS":
                long_fail += 1

        if direction == "TRADE":
            print(f"{entry_date} @ {entry_time} ---> {exit_date} @ {exit_time} ___ {result} __ {direction}")
        success_flag = 0
        result = "NO RESULT"

    # === Print Result Table ===
    print(f"{'Time':>5} | {'Successes':>9} | {'Failures':>8} | {'Success Rate':>13}")
    print("-" * 70)
    total_success = total_fail = total_effective_success = 0.0

    for slot in all_slots:
        success = time_bins[slot]["success"]
        fail = time_bins[slot]["fail"]
        total = success + fail
        rate = (success / total) * 100 if total > 0 else None
        total_success += success
        total_fail += fail
        rate_str = f"{rate:.2f}%" if rate is not None else "  N/A"
        print(f"{slot:>5} | {success:>9} | {fail:>8} | {rate_str:>12}")


    
    print("\n" + "-" * 70)
    print(f"Simulation period: {start_date} to {end_date} ({num_days} days)")
    print(f"Trades per day: {total / num_days:.2f}")
    print(f"Long Trades: {long_success + long_fail}, Long Success: {long_success}, Long Fail: {long_fail}")
    print(f"Short Trades: {short_success + short_fail}, Short Success: {short_success}, Short Fail: {short_fail}")
    print("=============================")
    print(f"OVERALL LONG WIN RATE: {round(100 * long_success/(long_success + long_fail + 0.001), 2)}%")
    print("=============================")
    print(f"Long: {long},  Short: {short},  No-trade: {no_trade},  total: {long+short+no_trade}")
"""