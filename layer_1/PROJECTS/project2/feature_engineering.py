"""
feature_engineering.py - Real-Time Streaming Feature Extraction & Temporal Splitting.

Why this file exists:
In production risk engineering, features must capture streaming velocity, spatial dynamics,
historical deviations from user baselines, and circadian patterns without data leakage.
"""

from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

from .config import HIGH_RISK_MERCHANTS, RANDOM_SEED
from .math_stats_engine import calculate_travel_velocity_kmh, haversine_distance_km


# =====================================================================
# 1. CANONICAL FEATURE COLUMN DEFINITIONS
# =====================================================================

FEATURE_COLUMNS: List[str] = [
    # 1. Amount & Deviation Signals
    "amount",
    "amount_to_user_avg_ratio",
    "amount_zscore_user",
    "amount_first_digit",
    
    # 2. Time-Windowed Velocity Signals
    "velocity_count_1m",
    "velocity_count_10m",
    "velocity_count_1h",
    "velocity_count_24h",
    "velocity_spend_sum_10m",
    "velocity_spend_sum_1h",
    "velocity_spend_sum_24h",
    
    # 3. Spatial & Kinematic Telemetry
    "distance_from_last_tx_km",
    "travel_speed_kmh",
    "is_impossible_travel",
    
    # 4. Device & Identity Telemetry
    "is_new_device",
    "device_count_24h",
    
    # 5. Temporal & Circadian Signals
    "hour_of_day",
    "hour_sin",
    "hour_cos",
    "is_night_hours",
    "is_weekend",
    
    # 6. Domain Risk Signals
    "is_international",
    "is_high_risk_merchant",
    "is_high_risk_channel",
]

TARGET_COLUMN: str = "is_fraud"


# =====================================================================
# 2. STREAMING & HISTORICAL FEATURE EXTRACTION ENGINE
# =====================================================================

def extract_features_from_transactions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transforms raw chronological transaction logs into a rich feature matrix.
    Processes user transactions chronologically to simulate real-time feature stores.
    """
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(by="timestamp").reset_index(drop=True)
    
    feature_rows: List[Dict] = []
    
    # Group by user to maintain isolated state per cardholder
    grouped_users = df.groupby("user_id")
    
    for user_id, user_df in grouped_users:
        user_df = user_df.sort_values(by="timestamp")
        
        seen_devices = set()
        history_amounts: List[float] = []
        history_timestamps: List[pd.Timestamp] = []
        history_lats: List[float] = []
        history_lons: List[float] = []
        history_devices: List[str] = []
        
        for idx, row in user_df.iterrows():
            curr_time: pd.Timestamp = row["timestamp"]
            curr_epoch: float = curr_time.timestamp()
            curr_amount: float = float(row["amount"])
            curr_lat: float = float(row["latitude"])
            curr_lon: float = float(row["longitude"])
            curr_device: str = str(row["device_id"])
            curr_channel: str = str(row["channel"])
            curr_merchant: str = str(row["merchant_category"])
            
            # ---------------------------------------------------------
            # A. Amount Baseline Deviations
            # ---------------------------------------------------------
            if len(history_amounts) >= 3:
                user_avg = np.mean(history_amounts)
                user_std = np.std(history_amounts) + 1e-4
                amount_to_avg_ratio = curr_amount / (user_avg + 1.0)
                amount_zscore = (curr_amount - user_avg) / user_std
            else:
                amount_to_avg_ratio = 1.0
                amount_zscore = 0.0
                
            # First leading digit (for Benford's Law analysis)
            first_digit = int(str(f"{curr_amount:.6e}")[0])
            
            # ---------------------------------------------------------
            # B. Streaming Time-Windowed Velocity Counters
            # ---------------------------------------------------------
            v_1m_cnt = 0
            v_10m_cnt = 0
            v_1h_cnt = 0
            v_24h_cnt = 0
            v_10m_spend = 0.0
            v_1h_spend = 0.0
            v_24h_spend = 0.0
            
            devices_24h = set()
            
            for past_amt, past_time, past_dev in zip(reversed(history_amounts), reversed(history_timestamps), reversed(history_devices)):
                delta_sec = (curr_time - past_time).total_seconds()
                
                if delta_sec > 86400.0:  # Beyond 24 hours
                    break
                    
                if delta_sec <= 86400.0: # 24 Hours
                    v_24h_cnt += 1
                    v_24h_spend += past_amt
                    devices_24h.add(past_dev)
                    
                if delta_sec <= 3600.0:  # 1 Hour
                    v_1h_cnt += 1
                    v_1h_spend += past_amt
                    
                if delta_sec <= 600.0:   # 10 Minutes
                    v_10m_cnt += 1
                    v_10m_spend += past_amt
                    
                if delta_sec <= 60.0:    # 1 Minute
                    v_1m_cnt += 1
                    
            devices_24h.add(curr_device)
            device_count_24h = len(devices_24h)
            
            # ---------------------------------------------------------
            # C. Spatial & Kinematic Telemetry
            # ---------------------------------------------------------
            if len(history_lats) > 0:
                last_lat = history_lats[-1]
                last_lon = history_lons[-1]
                last_epoch = history_timestamps[-1].timestamp()
                
                dist_km = haversine_distance_km(last_lat, last_lon, curr_lat, curr_lon)
                speed_kmh = calculate_travel_velocity_kmh(last_lat, last_lon, last_epoch, curr_lat, curr_lon, curr_epoch)
            else:
                dist_km = 0.0
                speed_kmh = 0.0
                
            is_impossible = 1 if speed_kmh > 900.0 else 0
            
            # ---------------------------------------------------------
            # D. Device Fingerprint Telemetry
            # ---------------------------------------------------------
            is_new_device = 1 if (curr_device not in seen_devices and len(seen_devices) > 0) else 0
            seen_devices.add(curr_device)
            
            # ---------------------------------------------------------
            # E. Temporal & Circadian Signals
            # ---------------------------------------------------------
            hour = curr_time.hour
            hour_sin = np.sin(2.0 * np.pi * hour / 24.0)
            hour_cos = np.cos(2.0 * np.pi * hour / 24.0)
            is_night = 1 if 0 <= hour <= 5 else 0
            is_weekend = 1 if curr_time.weekday() >= 5 else 0
            
            # ---------------------------------------------------------
            # F. Domain Risk Signals
            # ---------------------------------------------------------
            is_intl = int(row.get("is_international", 0))
            is_high_risk_merc = 1 if curr_merchant in HIGH_RISK_MERCHANTS else 0
            is_high_risk_chan = 1 if curr_channel in ["CRYPTO_RAMP", "INTERNATIONAL_WIRE"] else 0
            
            feat_record = {
                # Identifiers & Target
                "transaction_id": row["transaction_id"],
                "user_id": user_id,
                "timestamp": curr_time,
                "attack_type": row.get("attack_type", "UNKNOWN"),
                "is_fraud": int(row["is_fraud"]),
                
                # 1. Amount & Deviation
                "amount": curr_amount,
                "amount_to_user_avg_ratio": round(amount_to_avg_ratio, 4),
                "amount_zscore_user": round(amount_zscore, 4),
                "amount_first_digit": first_digit,
                
                # 2. Velocity
                "velocity_count_1m": v_1m_cnt,
                "velocity_count_10m": v_10m_cnt,
                "velocity_count_1h": v_1h_cnt,
                "velocity_count_24h": v_24h_cnt,
                "velocity_spend_sum_10m": round(v_10m_spend, 2),
                "velocity_spend_sum_1h": round(v_1h_spend, 2),
                "velocity_spend_sum_24h": round(v_24h_spend, 2),
                
                # 3. Spatial
                "distance_from_last_tx_km": round(dist_km, 2),
                "travel_speed_kmh": round(speed_kmh, 2),
                "is_impossible_travel": is_impossible,
                
                # 4. Device
                "is_new_device": is_new_device,
                "device_count_24h": device_count_24h,
                
                # 5. Temporal
                "hour_of_day": hour,
                "hour_sin": round(hour_sin, 4),
                "hour_cos": round(hour_cos, 4),
                "is_night_hours": is_night,
                "is_weekend": is_weekend,
                
                # 6. Domain Risk
                "is_international": is_intl,
                "is_high_risk_merchant": is_high_risk_merc,
                "is_high_risk_channel": is_high_risk_chan,
            }
            feature_rows.append(feat_record)
            
            # Update user memory
            history_amounts.append(curr_amount)
            history_timestamps.append(curr_time)
            history_lats.append(curr_lat)
            history_lons.append(curr_lon)
            history_devices.append(curr_device)
            
    feat_df = pd.DataFrame(feature_rows)
    feat_df = feat_df.sort_values(by="timestamp").reset_index(drop=True)
    return feat_df


# =====================================================================
# 3. LEAK-FREE TEMPORAL TRAIN / VAL / TEST SPLIT
# =====================================================================

def temporal_train_val_test_split(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Splits time-series data strictly chronologically:
      - Train Set: First 70% of historical timeline
      - Val Set: Next 15% of historical timeline
      - Test Set (Out-of-Time OOT): Final 15% of historical timeline
      
    Why random splitting is prohibited in ML risk systems:
    Random K-Fold splitting trains on future transactions (Day 150) to predict past
    transactions (Day 30), artificially inflating test metrics and leaking temporal trends.
    """
    df_sorted = df.sort_values(by="timestamp").reset_index(drop=True)
    n = len(df_sorted)
    
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    train_df = df_sorted.iloc[:train_end].copy().reset_index(drop=True)
    val_df = df_sorted.iloc[train_end:val_end].copy().reset_index(drop=True)
    test_df = df_sorted.iloc[val_end:].copy().reset_index(drop=True)
    
    return train_df, val_df, test_df
