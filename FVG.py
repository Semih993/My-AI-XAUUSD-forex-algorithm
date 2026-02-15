import pandas as pd

def add_fvg_features(df, max_zones=3, min_width=0.00150, max_reversals=1, penetration_threshold=0.15):
    """
    Detects Fair Value Gaps (FVGs) and appends FVG features to the dataframe.
    Zones are deactivated if price pierces through more than 15% or are used more than twice.

    Parameters:
        df (pd.DataFrame): Must contain 'High', 'Low', 'Close' columns.
        max_zones (int): Number of nearest FVG zones to include as features.
        min_width (float): Minimum gap width to consider.
        max_reversals (int): Max allowed reversals before a zone is invalidated.
        penetration_threshold (float): Allowed penetration into zone before invalidation (as a fraction of width).

    Returns:
        df (pd.DataFrame): Same as input but with extra FVG feature columns.
    """
    # -------- Step 1: Detect FVG Zones --------
    fvg_zones = []
    for i in range(len(df) - 2):
        high_i = df["High"].iloc[i]
        low_i2 = df["Low"].iloc[i + 2]
        low_i = df["Low"].iloc[i]
        high_i2 = df["High"].iloc[i + 2]

        # Bullish FVG
        if low_i2 > high_i:
            gap_top = low_i2
            gap_bottom = high_i
            width = gap_top - gap_bottom
            if width >= min_width:
                fvg_zones.append({
                    "type": "bullish",
                    "start_idx": i,
                    "zone_top": gap_top,
                    "zone_bottom": gap_bottom,
                    "width": width,
                    "timestamp": df.index[i + 2],
                    "reversals": 0
                })

        # Bearish FVG
        elif high_i2 < low_i:
            gap_top = low_i
            gap_bottom = high_i2
            width = gap_top - gap_bottom
            if width >= min_width:
                fvg_zones.append({
                    "type": "bearish",
                    "start_idx": i,
                    "zone_top": gap_top,
                    "zone_bottom": gap_bottom,
                    "width": width,
                    "timestamp": df.index[i + 2],
                    "reversals": 0
                })

    # -------- Step 2: Generate Features Per Row --------
    all_fvg_features = []
    active_zones = []

    for i in range(len(df)):
        current_time = df.index[i]
        current_price = df["Close"].iloc[i]
        candle_high = df["High"].iloc[i]
        candle_low = df["Low"].iloc[i]
        candle_close = df["Close"].iloc[i]

        # Update active zones with expiry rules
        updated_zones = []
        for zone in fvg_zones:
            if zone["timestamp"] >= current_time:
                continue  # future zone

            zone_top = zone["zone_top"]
            zone_bottom = zone["zone_bottom"]
            width = zone["width"]
            penetration_limit = penetration_threshold * width

            # Invalidation logic
            if zone["type"] == "bullish":
                penetration = max(0, zone_bottom - candle_low)
                if penetration > penetration_limit:
                    continue  # deeply penetrated = invalid
                elif candle_low <= zone_top and candle_close > zone_top:
                    zone["reversals"] += 1
                    if zone["reversals"] > max_reversals:
                        continue
            elif zone["type"] == "bearish":
                penetration = max(0, candle_high - zone_top)
                if penetration > penetration_limit:
                    continue
                elif candle_high >= zone_bottom and candle_close < zone_bottom:
                    zone["reversals"] += 1
                    if zone["reversals"] > max_reversals:
                        continue

            updated_zones.append(zone)

        active_zones = updated_zones  # update zone pool

        # -------- Step 3: Extract Zone Features for This Candle --------
        zone_features = []
        for zone in active_zones:
            zone_high = zone["zone_top"]
            zone_low = zone["zone_bottom"]
            width = zone["width"]
            midpoint = (zone_high + zone_low) / 2
            above_flag = 1 if current_price > midpoint else 0
            distance = abs(current_price - midpoint)

            zone_features.append({
                "high": zone_high,
                "low": zone_low,
                "width": width,
                "above": above_flag,
                "distance": distance
            })

        # Sort zones by proximity
        zone_features.sort(key=lambda z: z["distance"])

        # Take top N and flatten
        fvg_vector = []
        for j in range(max_zones):
            if j < len(zone_features):
                z = zone_features[j]
                fvg_vector.extend([z["high"], z["low"], z["width"], z["above"], z["distance"]])
            else:
                fvg_vector.extend([0, 0, 0, 0, 0])  # pad

        all_fvg_features.append(fvg_vector)

    # -------- Step 4: Add Features to DataFrame --------
    col_names = []
    for i in range(max_zones):
        col_names += [
            f"FVG_{i+1}_High", f"FVG_{i+1}_Low", f"FVG_{i+1}_Width",
            f"FVG_{i+1}_Above", f"FVG_{i+1}_Distance"
        ]

    fvg_feature_df = pd.DataFrame(all_fvg_features, columns=col_names, index=df.index)
    df = pd.concat([df, fvg_feature_df], axis=1)

    return df
