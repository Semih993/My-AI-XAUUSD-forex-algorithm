def advanced_galactic_backtest(flips, times, prices, model, device, atr, plus_di, minus_di,
                                min_confidence=0.8):

    if len(flips) == 0:
        print("No flips provided for backtest.")
        return

    close_prices = [c[3] for c in prices]
    real, imag = generate_hilbert_signals(close_prices)

    total_trades = 0
    wins = 0
    losses = 0
    undetermined = 0

    for flip_idx in flips:
        if flip_idx + 1 >= len(prices):
            continue
        if flip_idx - 15 < 0:
            continue

        X = prepare_live_input(prices, real, imag, atr, plus_di, minus_di, flip_idx)
        if X is None:
            continue

        X = X.to(device)
        prediction = torch.softmax(model(X), dim=1)
        model_direction = torch.argmax(prediction, dim=1).item()
        confidence = prediction[0][model_direction].item()

        if confidence < min_confidence:
            continue  # Skip low-confidence trades

        # Determine recent trend
        recent_closes = [prices[i][3] for i in range(flip_idx - 10, flip_idx)]
        trend_slope = (recent_closes[-1] - recent_closes[0]) / 10

        final_direction = model_direction

        next_open = prices[flip_idx + 1][0]
        next_close = prices[flip_idx + 1][3]

        # If model predicts with the trend, require strong confirmation
        if (model_direction == 1 and trend_slope > 0) or (model_direction == 0 and trend_slope < 0):
            next_body = abs(next_close - next_open)
            current_atr = atr[flip_idx + 1]

            if model_direction == 1 and next_close < next_open and next_body > 0.2 * current_atr:
                final_direction = 0  # Reject long if strong bearish candle
            elif model_direction == 0 and next_close > next_open and next_body > 0.2 * current_atr:
                final_direction = 1  # Reject short if strong bullish candle
            else:
                # If no strong confirmation, be reluctant and reverse the trade
                final_direction = 0 if model_direction == 1 else 1

        entry_time = times[flip_idx + 1]
        entry_price = prices[flip_idx + 1][0]

        if final_direction == 1:
            tp_level = entry_price * (1 + 0.003)
            sl_level = entry_price * (1 - 0.003)
        else:
            tp_level = entry_price * (1 + 0.003)
            sl_level = entry_price * (1 - 0.003)

        result = None

        for future_idx in range(flip_idx + 2, len(prices)):
            high_price = prices[future_idx][1]
            low_price = prices[future_idx][2]

            if final_direction == 1:
                if high_price >= tp_level:
                    result = 'TP'
                    wins += 1
                    break
                if low_price <= sl_level:
                    result = 'SL'
                    losses += 1
                    break
            else:
                if low_price <= tp_level:
                    result = 'TP'
                    wins += 1
                    break
                if high_price >= sl_level:
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
    print(f"\n====== Galactic Backtest Summary ======")
    print(f"Total Trades Executed: {total_trades}")
    print(f"Wins: {wins}")
    print(f"Losses: {losses}")
    print(f"Success Rate: {success_rate:.2f}%")

    return success_rate
