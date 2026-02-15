def backtest_with_delayed_entry(
    flips, times, prices, model, device, atr, plus_di, minus_di,
    min_confidence=0.8, min_di_diff=5, spike_multiplier=2, lookback=10):

    if len(flips) == 0:
        print("No flips provided for backtest.")
        return

    close_prices = [c[3] for c in prices]
    real, imag = generate_hilbert_signals(close_prices)

    total_trades = 0
    wins = 0
    losses = 0

    for flip_idx in flips:
        if flip_idx + 1 >= len(prices):  # Ensure the next candle exists
            continue
        if flip_idx - max(lookback, 15) < 0:
            continue

        X = prepare_live_input(prices, real, imag, atr, plus_di, minus_di, flip_idx)
        if X is None:
            continue

        X = X.to(device)
        prediction = torch.softmax(model(X), dim=1)
        model_direction = torch.argmax(prediction, dim=1).item()  # 1 = Long, 0 = Short
        confidence = prediction[0][model_direction].item()

        if confidence < min_confidence:
            continue  # Skip trades with low confidence

        # ===== Decision Logic =====
        final_direction = model_direction

        # ===== Wait for the next candle confirmation =====
        next_open = prices[flip_idx + 1][0]
        next_close = prices[flip_idx + 1][3]

        # If the next candle moves against the model, reverse
        if model_direction == 1 and next_close < next_open:
            final_direction = 0  # Reverse to short
        elif model_direction == 0 and next_close > next_open:
            final_direction = 1  # Reverse to long
        # Otherwise, keep model direction

        entry_time = times[flip_idx + 1]  # We enter at the open of the next candle
        entry_price = prices[flip_idx + 1][0]

        if final_direction == 1:  # Long
            tp_level = entry_price * (1 + 0.0030)
            sl_level = entry_price * (1 - 0.0030)
        else:  # Short
            tp_level = entry_price * (1 - 0.0030)
            sl_level = entry_price * (1 + 0.0030)

        result = None

        for future_idx in range(flip_idx + 2, len(prices)):
            open_price = prices[future_idx][0]
            close_price = prices[future_idx][3]

            if final_direction == 1:
                if open_price >= tp_level or close_price >= tp_level:
                    result = 'TP'
                    wins += 1
                    break
                if open_price <= sl_level or close_price <= sl_level:
                    result = 'SL'
                    losses += 1
                    break
            else:
                if open_price <= tp_level or close_price <= tp_level:
                    result = 'TP'
                    wins += 1
                    break
                if open_price >= sl_level or close_price >= sl_level:
                    result = 'SL'
                    losses += 1
                    break

        if result is None:
            final_close = prices[-1][3]
            if final_direction == 1 and final_close >= entry_price:
                wins += 1
                result = 'TP'
            elif final_direction == 0 and final_close <= entry_price:
                wins += 1
                result = 'TP'
            else:
                losses += 1
                result = 'SL'

        total_trades += 1

        direction_str = 'Long' if final_direction == 1 else 'Short'
        print(f"🕒 {entry_time} | Direction: {direction_str} | Result: {result}")

    success_rate = 100 * wins / total_trades if total_trades > 0 else 0

    print(f"\n====== Delayed Entry Backtest Summary ======")
    print(f"Total Trades Executed: {total_trades}")
    print(f"Wins: {wins}")
    print(f"Losses: {losses}")
    print(f"Success Rate: {success_rate:.2f}%")

    return success_rate
