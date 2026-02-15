import numpy as np
import pandas as pd

def wick_liquidity(
    df,
    max_zones=3,
    min_wick_size=0.00140,
    max_reversals=2,
    penetration_threshold=0.15
):
    """
    Detects liquidity wick zones where the wick is greater than or equal to the candle body.
    Zones are deactivated if price pierces too deep or gets respected too often.

    Parameters:
        df (pd.DataFrame): Must contain 'Open', 'High', 'Low', 'Close'
        max_zones (int): Max number of closest wick zones to track
        min_wick_size (float): Minimum wick size (absolute) to consider
        max_reversals (int): Max number of respected touches before invalidation
        penetration_threshold (float): % of wick size allowed to be pierced before invalidation

    Returns:
        pd.DataFrame: Original df with wick liquidity features appended
    """
    wick_zones = []

    for i in range(len(df)):
        open_ = df["Open"].iloc[i]
        close = df["Close"].iloc[i]
        high = df["High"].iloc[i]
        low = df["Low"].iloc[i]

        body_size = abs(close - open_)
        upper_wick = high - max(open_, close)
        lower_wick = min(open_, close) - low

        # Bullish wick zone
        if lower_wick >= upper_wick and lower_wick >= body_size and lower_wick >= min_wick_size:
            wick_zones.append({
                "type": "bullish",
                "level": low,
                "width": lower_wick,
                "timestamp": df.index[i],
                "reversals": 0
            })

        # Bearish wick zone
        elif upper_wick > lower_wick and upper_wick >= body_size and upper_wick >= min_wick_size:
            wick_zones.append({
                "type": "bearish",
                "level": high,
                "width": upper_wick,
                "timestamp": df.index[i],
                "reversals": 0
            })

    all_features = []
    active_zones = []

    for i in range(len(df)):
        current_time = df.index[i]
        current_price = df["Close"].iloc[i]
        high = df["High"].iloc[i]
        low = df["Low"].iloc[i]
        close = df["Close"].iloc[i]

        updated_zones = []
        for zone in wick_zones:
            if zone["timestamp"] >= current_time:
                continue

            level = zone["level"]
            width = zone["width"]
            penetration_limit = penetration_threshold * width

            if zone["type"] == "bullish":
                if low < (level - penetration_limit):
                    continue
                elif low <= level and close > level:
                    zone["reversals"] += 1
                    if zone["reversals"] > max_reversals:
                        continue
            elif zone["type"] == "bearish":
                if high > (level + penetration_limit):
                    continue
                elif high >= level and close < level:
                    zone["reversals"] += 1
                    if zone["reversals"] > max_reversals:
                        continue

            updated_zones.append(zone)

        active_zones = updated_zones

        zone_features = []
        for zone in active_zones:
            distance = abs(current_price - zone["level"])
            above = 1 if current_price > zone["level"] else 0
            zone_features.append({
                "level": zone["level"],
                "width": zone["width"],
                "above": above,
                "distance": distance
            })

        zone_features.sort(key=lambda z: z["distance"])
        wick_vector = []
        for j in range(max_zones):
            if j < len(zone_features):
                z = zone_features[j]
                wick_vector.extend([z["level"], z["width"], z["above"], z["distance"]])
            else:
                wick_vector.extend([0, 0, 0, 0])

        all_features.append(wick_vector)

    col_names = []
    for i in range(max_zones):
        col_names += [
            f"WICK_{i+1}_Level", f"WICK_{i+1}_Width",
            f"WICK_{i+1}_Above", f"WICK_{i+1}_Distance"
        ]

    wick_df = pd.DataFrame(all_features, columns=col_names, index=df.index)
    df = pd.concat([df, wick_df], axis=1)

    return df