"""
baselines.py - Heuristic Rule Engines & Dummy Baselines.

Why this file exists:
In ML engineering, we must always benchmark sophisticated ML/DL models against:
  1. The "Naive Majority Baseline" (demonstrating why 99.2% accuracy is a trap).
  2. The "Domain Heuristic Rule Engine" (traditional bank rule engine).
"""

from typing import Dict, List, Optional
import numpy as np
import pandas as pd


# =====================================================================
# 1. MAJORITY CLASS DUMMY BASELINE (THE ACCURACY PARADOX)
# =====================================================================

class MajorityClassBaseline:
    """
    Trivially predicts the majority class (0 = Legitimate) for 100% of transactions.
    
    Why we include this:
    On an imbalanced dataset with 99.2% legitimate data, this model scores 99.2% accuracy.
    However, its Recall on fraud is exactly 0.0, and fraud financial losses are 100%.
    It serves as the definitive proof that Accuracy is the wrong metric for risk systems.
    """
    def __init__(self):
        self.classes_ = np.array([0, 1])
        
    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "MajorityClassBaseline":
        return self
        
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.zeros(len(X), dtype=int)
        
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        # Returns [P(Legit)=1.0, P(Fraud)=0.0]
        n = len(X)
        proba = np.zeros((n, 2), dtype=float)
        proba[:, 0] = 1.0
        return proba


# =====================================================================
# 2. EXPERT DOMAIN HEURISTIC RULE ENGINE
# =====================================================================

class RuleBasedFraudEngine:
    """
    Expert system implementing classic financial fraud rules.
    
    Rules evaluated:
      - Rule 1 (Kinematic): Impossible travel speed (>900 km/h)
      - Rule 2 (Account Takeover): Midnight hours + New device + Amount > 4x User Average
      - Rule 3 (Bot Velocity): Rapid burst (>= 3 transactions in 1 minute)
      - Rule 4 (Structuring/Smurfing): Amount between ₹48,000 and ₹49,999 on high-risk channel
      - Rule 5 (High Risk Surge): International + High risk merchant + Amount > 6x User Average
    """
    def __init__(self):
        self.classes_ = np.array([0, 1])
        
    def fit(self, X: pd.DataFrame, y: Optional[np.ndarray] = None) -> "RuleBasedFraudEngine":
        # Rule engines are deterministic and do not require gradient-based training
        return self
        
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Calculates heuristic risk score based on rule violations.
        """
        df = X.copy()
        n = len(df)
        risk_scores = np.zeros(n, dtype=float)
        
        # Rule 1: Impossible travel (+0.85 risk)
        if "is_impossible_travel" in df.columns:
            risk_scores += np.where(df["is_impossible_travel"] == 1, 0.85, 0.0)
            
        # Rule 2: Midnight + New Device + High Spend (+0.75 risk)
        if all(col in df.columns for col in ["is_night_hours", "is_new_device", "amount_to_user_avg_ratio"]):
            ato_cond = (
                (df["is_night_hours"] == 1)
                & (df["is_new_device"] == 1)
                & (df["amount_to_user_avg_ratio"] > 4.0)
            )
            risk_scores += np.where(ato_cond, 0.75, 0.0)
            
        # Rule 3: Velocity Bot Storm (+0.80 risk)
        if "velocity_count_1m" in df.columns:
            risk_scores += np.where(df["velocity_count_1m"] >= 3, 0.80, 0.0)
            
        # Rule 4: Structuring / Smurfing (+0.70 risk)
        if all(col in df.columns for col in ["amount", "is_high_risk_channel"]):
            smurf_cond = (
                (df["amount"] >= 48000.0)
                & (df["amount"] <= 49999.0)
                & (df["is_high_risk_channel"] == 1)
            )
            risk_scores += np.where(smurf_cond, 0.70, 0.0)
            
        # Rule 5: High Risk Merchant Surge (+0.60 risk)
        if all(col in df.columns for col in ["is_high_risk_merchant", "amount_to_user_avg_ratio", "is_international"]):
            surge_cond = (
                (df["is_high_risk_merchant"] == 1)
                & (df["amount_to_user_avg_ratio"] > 6.0)
            )
            risk_scores += np.where(surge_cond, 0.60, 0.0)
            
        # Base ambient prior probability for unflagged transactions
        risk_scores = np.clip(risk_scores + 0.005, 0.0, 1.0)
        
        proba = np.zeros((n, 2), dtype=float)
        proba[:, 1] = risk_scores
        proba[:, 0] = 1.0 - risk_scores
        return proba
        
    def predict(self, X: pd.DataFrame, threshold: float = 0.50) -> np.ndarray:
        proba = self.predict_proba(X)
        return (proba[:, 1] >= threshold).astype(int)
