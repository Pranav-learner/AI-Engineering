"""
config.py - Domain Constants, Fraud Attack Taxonomy, User Personas, & Decision Tiers.

Why this file exists:
In production risk systems, configurations must be centralized, immutable, and reproducible.
This file defines baseline user behavioral profiles, adversarial attack signatures,
business financial loss matrices, and automated decision-routing thresholds.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple

# =====================================================================
# 1. GLOBAL REPRODUCIBILITY & SIMULATION CONSTANTS
# =====================================================================

# Global random seed ensures all stochastic data generation and model training runs
# are 100% deterministic and reproducible across team members and CI/CD pipelines.
RANDOM_SEED: int = 42

# Default length of synthetic transaction history (180 days = ~6 months)
DEFAULT_SIMULATION_DAYS: int = 180

# Number of distinct cardholders/users in our simulated cohort
NUM_SIMULATED_USERS: int = 40

# Target fraud prevalence (~0.8% of transactions are fraudulent, mirroring real-world credit card data)
TARGET_FRAUD_RATIO: float = 0.008


# =====================================================================
# 2. TRANSACTION CHANNELS & MERCHANT CATEGORIES
# =====================================================================

# Transaction channels define the physical or digital medium of payment
TRANSACTION_CHANNELS: List[str] = [
    "E_COMMERCE",         # Card-not-present online checkout (highest organic fraud risk)
    "POS_RETAIL",          # Physical chip/contactless terminal in stores
    "ATM_WITHDRAWAL",     # Physical cash withdrawal
    "P2P_TRANSFER",       # Peer-to-peer wallet transfer (UPI / Venmo / Zelle)
    "CRYPTO_RAMP",        # High-risk fiat-to-crypto gateway
    "INTERNATIONAL_WIRE", # Cross-border SWIFT/Wire transfer
]

# Standard Merchant Category Codes (MCC) representing spending domains
MERCHANT_CATEGORIES: List[str] = [
    "Groceries",
    "Dining_Restaurants",
    "Electronics_Tech",
    "Travel_Airlines",
    "Fuel_Gas",
    "Entertainment_Streaming",
    "Luxury_Jewelry",     # High fraud target
    "Gambling_Casinos",    # High fraud target
    "Crypto_Exchange",    # High fraud target
    "Pharmacy_Health",
]

# High-risk merchant categories that warrant elevated prior probability of fraud
HIGH_RISK_MERCHANTS: List[str] = [
    "Luxury_Jewelry",
    "Gambling_Casinos",
    "Crypto_Exchange",
]


# =====================================================================
# 3. FRAUD ATTACK TAXONOMY (ADVERSARIAL SIGNATURES)
# =====================================================================

class FraudAttackType:
    """
    Standard industry classification of financial attack vectors.
    """
    LEGITIMATE = "LEGITIMATE"
    ACCOUNT_TAKEOVER = "ACCOUNT_TAKEOVER"        # Stolen credentials, new device, midnight large transfer
    CARD_TESTING_BURST = "CARD_TESTING_BURST"    # Bot storm trying small rapid charges to test stolen card
    IMPOSSIBLE_TRAVEL = "IMPOSSIBLE_TRAVEL"      # Geographically impossible distance in short time
    STRUCTURING_SMURFING = "STRUCTURING_SMURFING"# Deliberately staying just below regulatory reporting thresholds
    HIGH_RISK_SURGE = "HIGH_RISK_SURGE"          # Sudden burst of luxury / crypto purchases on dormant card


# =====================================================================
# 4. USER BEHAVIORAL PERSONAS
# =====================================================================

@dataclass(frozen=True)
class UserPersonaConfig:
    """
    Defines baseline spending habits for a consumer segment.
    `frozen=True` ensures configs are immutable and cannot be mutated at runtime.
    """
    persona_name: str
    mean_amount: float            # Expected value (mean) of transaction amount
    amount_std: float             # Standard deviation of transaction amount
    daily_tx_rate: float          # Average Poisson arrival rate of transactions per day
    primary_channels: List[str]   # Usual channels utilized by this persona
    home_lat: float               # Home latitude coordinate
    home_lon: float               # Home longitude coordinate
    international_prob: float     # Likelihood of cross-border transactions


USER_PERSONAS: Dict[str, UserPersonaConfig] = {
    "college_student": UserPersonaConfig(
        persona_name="College Student",
        mean_amount=350.0,
        amount_std=200.0,
        daily_tx_rate=3.5,        # Frequent small purchases: canteen, coffee, streaming
        primary_channels=["E_COMMERCE", "P2P_TRANSFER", "POS_RETAIL"],
        home_lat=12.9716,         # Bangalore
        home_lon=77.5946,
        international_prob=0.01,
    ),
    "corporate_professional": UserPersonaConfig(
        persona_name="Corporate Professional",
        mean_amount=2200.0,
        amount_std=1500.0,
        daily_tx_rate=2.2,        # Groceries, dining out, fuel, weekend shopping
        primary_channels=["POS_RETAIL", "E_COMMERCE", "P2P_TRANSFER"],
        home_lat=19.0760,         # Mumbai
        home_lon=72.8777,
        international_prob=0.04,
    ),
    "business_owner_hni": UserPersonaConfig(
        persona_name="High-Net-Worth Business Owner",
        mean_amount=45000.0,
        amount_std=35000.0,
        daily_tx_rate=1.8,        # Large inventory payments, international wire, luxury
        primary_channels=["INTERNATIONAL_WIRE", "E_COMMERCE", "POS_RETAIL"],
        home_lat=28.6139,         # New Delhi
        home_lon=77.2090,
        international_prob=0.15,
    ),
    "senior_citizen_retiree": UserPersonaConfig(
        persona_name="Senior Citizen Retiree",
        mean_amount=1200.0,
        amount_std=600.0,
        daily_tx_rate=0.8,        # Low velocity, pharmacy, utility bills, POS at local stores
        primary_channels=["POS_RETAIL", "ATM_WITHDRAWAL"],
        home_lat=13.0827,         # Chennai
        home_lon=80.2707,
        international_prob=0.005,
    ),
}


# =====================================================================
# 5. BUSINESS COST MATRIX & DECISION TIERS
# =====================================================================

# In fraud detection, asymmetric costs dictate the optimal decision threshold:
# - False Positive (FP): Flagging a legitimate transaction costs customer friction,
#   potential customer churn, and manual support investigation (~₹250).
# - False Negative (FN): Missing a real fraud transaction costs the total stolen money
#   plus bank chargeback dispute penalty fees (~₹1,500 + stolen amount).
COST_FALSE_POSITIVE: float = 250.0
COST_FALSE_NEGATIVE_BASE: float = 1500.0

class DecisionTier:
    """
    Three-way operational decision routing for real-time payment gateways.
    """
    ALLOW = 0         # P(Fraud) < THRESHOLD_CHALLENGE -> Approve immediately (< 5ms)
    CHALLENGE = 1     # THRESHOLD_CHALLENGE <= P(Fraud) < THRESHOLD_BLOCK -> Step-up 2FA (OTP/Biometric)
    BLOCK = 2         # P(Fraud) >= THRESHOLD_BLOCK -> Reject transaction & freeze card

# Probability thresholds for the decision engine
THRESHOLD_CHALLENGE: float = 0.20
THRESHOLD_BLOCK: float = 0.75

DECISION_TIER_LABELS: Dict[int, str] = {
    DecisionTier.ALLOW: "🟢 ALLOW (Frictionless Approval)",
    DecisionTier.CHALLENGE: "🟡 CHALLENGE (Step-Up 2FA / Biometric)",
    DecisionTier.BLOCK: "🔴 BLOCK (High Risk - Card Freeze)",
}
