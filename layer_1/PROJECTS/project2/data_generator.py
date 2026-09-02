"""
data_generator.py creates a realistic time-series transaction stream spanning 180 days across multi-user cohorts. It models:

Circadian Diurnal Cycles: Inhomogeneous Poisson arrival process (high shopping activity 10:00–22:00, low organic activity 01:00–05:00).
Heavy-Tailed Spending Distributions: Log-Normal distributions tailored to each user's financial persona.
5 Realistic Adversarial Fraud Attacks:
🔴 Account Takeover: Midnight login + brand new device + high-value transfer.
🔴 Card Testing Burst: Automated bot storm firing 5–10 micro-transactions ($< ₹150$) within 60 seconds.
🔴 Impossible Travel: Super-speed geolocation jumps ($> 2,000\text{ km/h}$) between consecutive swipes.
🔴 Structuring / Smurfing: Deliberately splitting large stolen funds into multiple transactions just below AML reporting thresholds (e.g. ₹48,500 – ₹49,800).
🔴 High-Risk Surge: Sudden high-value Crypto / Luxury / Casino purchases on a conservative card.
"""

"""
data_generator.py - Realistic Financial Transaction & Adversarial Attack Simulator.

Why this file exists:
Fraud detection requires realistic time-series data reflecting circadian rhythms,
user persona baselines, device fingerprints, geographic coordinates, and targeted
adversarial attacks with distinct structural signatures.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from .config import (
    DEFAULT_SIMULATION_DAYS,
    HIGH_RISK_MERCHANTS,
    MERCHANT_CATEGORIES,
    NUM_SIMULATED_USERS,
    RANDOM_SEED,
    TARGET_FRAUD_RATIO,
    TRANSACTION_CHANNELS,
    USER_PERSONAS,
    FraudAttackType,
    UserPersonaConfig,
)


# =====================================================================
# 1. DIURNAL (CIRCADIAN) HOURLY PROBABILITY WEIGHTS
# =====================================================================

# Probability weights for each hour of the day (0 to 23).
# Legitimate human spending peaks during the day/evening and drops during sleep hours.
HOURLY_ACTIVITY_WEIGHTS: np.ndarray = np.array([
    0.01, 0.005, 0.005, 0.005, 0.01, 0.02,  # 00:00 - 05:59 (Deep Night - very low)
    0.04, 0.06, 0.08, 0.09, 0.08, 0.08,     # 06:00 - 11:59 (Morning surge)
    0.08, 0.08, 0.07, 0.07, 0.08, 0.09,     # 12:00 - 17:59 (Afternoon activity)
    0.10, 0.11, 0.09, 0.06, 0.04, 0.02,     # 18:00 - 23:59 (Evening shopping / dining peak)
])
HOURLY_ACTIVITY_WEIGHTS = HOURLY_ACTIVITY_WEIGHTS / np.sum(HOURLY_ACTIVITY_WEIGHTS)


# =====================================================================
# 2. SINGLE USER ORGANIC STREAM GENERATOR
# =====================================================================

def generate_user_organic_transactions(
    user_id: str,
    persona_key: str,
    start_date: datetime,
    num_days: int = DEFAULT_SIMULATION_DAYS,
    rng: Optional[np.random.RandomState] = None,
) -> List[Dict]:
    """
    Generates realistic, benign (legitimate) transactions for a specific user persona.
    
    Why Log-Normal for amounts:
    Financial spending amounts are non-negative, right-skewed, and heavy-tailed.
    Log-Normal(mu_log, sigma_log) accurately models standard living purchases with rare organic large spikes.
    """
    if rng is None:
        rng = np.random.RandomState(RANDOM_SEED)
        
    persona: UserPersonaConfig = USER_PERSONAS[persona_key]
    transactions: List[Dict] = []
    
    # Calculate Log-Normal parameters from desired mean and standard deviation:
    #   sigma_log^2 = ln(1 + (std / mean)^2)
    #   mu_log = ln(mean) - (sigma_log^2 / 2)
    variance_ratio = (persona.amount_std / persona.mean_amount) ** 2
    sigma_log = np.sqrt(np.log(1.0 + variance_ratio))
    mu_log = np.log(persona.mean_amount) - (0.5 * (sigma_log ** 2))
    
    # Known primary and secondary devices for this user
    user_devices = [f"DEV_{user_id}_PRIMARY", f"DEV_{user_id}_LAPTOP"]
    
    for day in range(num_days):
        current_day_date = start_date + timedelta(days=day)
        
        # Daily transaction count drawn from Poisson distribution based on persona rate
        num_tx_today = rng.poisson(lam=persona.daily_tx_rate)
        if num_tx_today == 0:
            continue
            
        for _ in range(num_tx_today):
            # Sample hour based on human circadian weights
            hour = rng.choice(24, p=HOURLY_ACTIVITY_WEIGHTS)
            minute = rng.randint(0, 60)
            second = rng.randint(0, 60)
            tx_time = current_day_date.replace(hour=hour, minute=minute, second=second)
            
            # Sample amount from persona Log-Normal distribution (minimum transaction ₹10)
            amount = max(10.0, float(rng.lognormal(mean=mu_log, sigma=sigma_log)))
            
            # Sample channel and merchant category
            channel = rng.choice(persona.primary_channels)
            merchant = rng.choice(MERCHANT_CATEGORIES)
            
            # Device: 90% primary phone, 10% secondary laptop
            device_id = rng.choice(user_devices, p=[0.90, 0.10])
            
            # Geo location: Gaussian jitter around home coordinates (~2 to 15 km radius)
            lat = persona.home_lat + rng.normal(0.0, 0.03)
            lon = persona.home_lon + rng.normal(0.0, 0.03)
            
            is_international = int(rng.rand() < persona.international_prob)
            if is_international:
                # Occasional foreign vacation / business trip
                lat += rng.choice([-15.0, 25.0])
                lon += rng.choice([-40.0, 50.0])
                
            tx_record = {
                "transaction_id": f"TX_{user_id}_{len(transactions):06d}",
                "user_id": user_id,
                "timestamp": tx_time,
                "amount": round(amount, 2),
                "channel": channel,
                "merchant_category": merchant,
                "device_id": device_id,
                "latitude": round(lat, 4),
                "longitude": round(lon, 4),
                "is_international": is_international,
                "is_fraud": 0,
                "attack_type": FraudAttackType.LEGITIMATE,
                "user_persona": persona.persona_name,
            }
            transactions.append(tx_record)
            
    return transactions


# =====================================================================
# 3. ADVERSARIAL FRAUD ATTACK INJECTOR
# =====================================================================

def inject_fraud_attacks(
    transactions: List[Dict],
    persona_key: str,
    rng: np.random.RandomState,
) -> List[Dict]:
    """
    Injects realistic adversarial attacks with specific structural fraud patterns.
    """
    if len(transactions) < 20:
        return transactions
        
    persona = USER_PERSONAS[persona_key]
    user_id = transactions[0]["user_id"]
    
    # Decide which attack vector to launch for this user
    attack_type = rng.choice([
        FraudAttackType.ACCOUNT_TAKEOVER,
        FraudAttackType.CARD_TESTING_BURST,
        FraudAttackType.IMPOSSIBLE_TRAVEL,
        FraudAttackType.STRUCTURING_SMURFING,
        FraudAttackType.HIGH_RISK_SURGE,
    ])
    
    # Pick an insertion point in the second half of history (after baseline is established)
    idx = rng.randint(len(transactions) // 2, len(transactions) - 10)
    base_tx = transactions[idx]
    base_time = base_tx["timestamp"]
    
    injected: List[Dict] = []
    
    # -----------------------------------------------------------------
    # ATTACK 1: ACCOUNT TAKEOVER (ATO)
    # Stolen credentials used on a new device during deep night for large wire/transfer
    # -----------------------------------------------------------------
    if attack_type == FraudAttackType.ACCOUNT_TAKEOVER:
        ato_time = base_time.replace(hour=rng.choice([1, 2, 3, 4]), minute=rng.randint(5, 55))
        amount = persona.mean_amount * rng.uniform(6.0, 15.0) # Massive spike
        
        injected.append({
            "transaction_id": f"TX_{user_id}_FRAUD_ATO",
            "user_id": user_id,
            "timestamp": ato_time,
            "amount": round(amount, 2),
            "channel": "P2P_TRANSFER" if rng.rand() < 0.5 else "INTERNATIONAL_WIRE",
            "merchant_category": "Crypto_Exchange",
            "device_id": f"DEV_UNKNOWN_{rng.randint(1000, 9999)}", # Brand new unfamiliar device
            "latitude": persona.home_lat + rng.normal(5.0, 1.0),
            "longitude": persona.home_lon + rng.normal(5.0, 1.0),
            "is_international": 1,
            "is_fraud": 1,
            "attack_type": FraudAttackType.ACCOUNT_TAKEOVER,
            "user_persona": persona.persona_name,
        })
        
    # -----------------------------------------------------------------
    # ATTACK 2: CARD TESTING BURST
    # Bot script attempting rapid micro-charges to verify stolen card number
    # -----------------------------------------------------------------
    elif attack_type == FraudAttackType.CARD_TESTING_BURST:
        num_bursts = rng.randint(5, 9)
        burst_start = base_time + timedelta(hours=rng.randint(1, 6))
        for b in range(num_bursts):
            # Rapid micro-transactions spaced 10 to 30 seconds apart
            b_time = burst_start + timedelta(seconds=b * rng.randint(10, 30))
            b_amount = rng.uniform(20.0, 140.0) # Tiny probe amount
            
            injected.append({
                "transaction_id": f"TX_{user_id}_FRAUD_BURST_{b}",
                "user_id": user_id,
                "timestamp": b_time,
                "amount": round(b_amount, 2),
                "channel": "E_COMMERCE",
                "merchant_category": "Electronics_Tech",
                "device_id": f"DEV_BOT_EMULATOR_{rng.randint(100, 999)}",
                "latitude": 37.7749, # US IP address proxy
                "longitude": -122.4194,
                "is_international": 1,
                "is_fraud": 1,
                "attack_type": FraudAttackType.CARD_TESTING_BURST,
                "user_persona": persona.persona_name,
            })
            
    # -----------------------------------------------------------------
    # ATTACK 3: IMPOSSIBLE TRAVEL
    # Card used at home location, then 30 mins later in London (>5,000 km/h)
    # -----------------------------------------------------------------
    elif attack_type == FraudAttackType.IMPOSSIBLE_TRAVEL:
        travel_time = base_time + timedelta(minutes=rng.randint(20, 45))
        injected.append({
            "transaction_id": f"TX_{user_id}_FRAUD_IMPOSSIBLE_TRAVEL",
            "user_id": user_id,
            "timestamp": travel_time,
            "amount": round(persona.mean_amount * rng.uniform(2.0, 5.0), 2),
            "channel": "POS_RETAIL",
            "merchant_category": "Luxury_Jewelry",
            "device_id": f"DEV_OVERSEAS_TERMINAL_{rng.randint(100, 999)}",
            "latitude": 51.5074, # London GPS coords (~7,000 km away from India)
            "longitude": -0.1278,
            "is_international": 1,
            "is_fraud": 1,
            "attack_type": FraudAttackType.IMPOSSIBLE_TRAVEL,
            "user_persona": persona.persona_name,
        })
        
    # -----------------------------------------------------------------
    # ATTACK 4: STRUCTURING / SMURFING
    # Multiple transactions engineered just under regulatory radar (₹48,000 - ₹49,900)
    # -----------------------------------------------------------------
    elif attack_type == FraudAttackType.STRUCTURING_SMURFING:
        num_smurfs = rng.randint(3, 5)
        for s in range(num_smurfs):
            s_time = base_time + timedelta(minutes=s * rng.randint(15, 45))
            # Staying just under ₹50,000 threshold
            s_amount = rng.uniform(48500.0, 49850.0)
            
            injected.append({
                "transaction_id": f"TX_{user_id}_FRAUD_SMURF_{s}",
                "user_id": user_id,
                "timestamp": s_time,
                "amount": round(s_amount, 2),
                "channel": "P2P_TRANSFER",
                "merchant_category": "Crypto_Exchange",
                "device_id": f"DEV_PROXY_{rng.randint(10, 99)}",
                "latitude": persona.home_lat + rng.normal(0.0, 0.05),
                "longitude": persona.home_lon + rng.normal(0.0, 0.05),
                "is_international": 0,
                "is_fraud": 1,
                "attack_type": FraudAttackType.STRUCTURING_SMURFING,
                "user_persona": persona.persona_name,
            })
            
    # -----------------------------------------------------------------
    # ATTACK 5: HIGH-RISK SURGE
    # Sudden burst of luxury/gambling/crypto on dormant card
    # -----------------------------------------------------------------
    else:
        surge_time = base_time + timedelta(hours=rng.randint(2, 8))
        injected.append({
            "transaction_id": f"TX_{user_id}_FRAUD_SURGE",
            "user_id": user_id,
            "timestamp": surge_time,
            "amount": round(persona.mean_amount * rng.uniform(8.0, 20.0), 2),
            "channel": "CRYPTO_RAMP",
            "merchant_category": rng.choice(HIGH_RISK_MERCHANTS),
            "device_id": f"DEV_{user_id}_PRIMARY",
            "latitude": persona.home_lat,
            "longitude": persona.home_lon,
            "is_international": 1,
            "is_fraud": 1,
            "attack_type": FraudAttackType.HIGH_RISK_SURGE,
            "user_persona": persona.persona_name,
        })
        
    # Combine organic and injected fraud transactions
    all_txs = transactions + injected
    
    # Sort strictly chronologically by timestamp
    all_txs.sort(key=lambda x: x["timestamp"])
    return all_txs


# =====================================================================
# 4. COHORT DATASET GENERATOR (END-TO-END PIPELINE)
# =====================================================================

def generate_transaction_dataset(
    num_users: int = NUM_SIMULATED_USERS,
    num_days: int = DEFAULT_SIMULATION_DAYS,
    random_seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """
    Generates a complete, multi-user cohort dataset with realistic class imbalance (~0.8% fraud).
    
    Returns:
      Pandas DataFrame containing clean, chronological transaction stream logs.
    """
    rng = np.random.RandomState(random_seed)
    start_date = datetime(2026, 1, 1, 0, 0, 0)
    
    persona_keys = list(USER_PERSONAS.keys())
    all_records: List[Dict] = []
    
    # Track how many users will be targeted by fraud attacks to maintain ~0.8% ratio
    num_fraud_targets = max(2, int(num_users * 0.35))
    fraud_user_indices = set(rng.choice(num_users, size=num_fraud_targets, replace=False))
    
    for i in range(num_users):
        user_id = f"USR_{i+1:04d}"
        persona_key = persona_keys[i % len(persona_keys)]
        
        # 1. Generate organic history
        user_txs = generate_user_organic_transactions(
            user_id=user_id,
            persona_key=persona_key,
            start_date=start_date,
            num_days=num_days,
            rng=rng,
        )
        
        # 2. Inject adversarial attack if targeted
        if i in fraud_user_indices:
            user_txs = inject_fraud_attacks(user_txs, persona_key=persona_key, rng=rng)
            
        all_records.extend(user_txs)
        
    df = pd.DataFrame(all_records)
    
    # Sort entire dataset chronologically
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(by="timestamp").reset_index(drop=True)
    
    return df
