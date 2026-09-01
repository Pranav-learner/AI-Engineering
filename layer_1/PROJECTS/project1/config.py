"""
config.py - Central configuration and domain constants for the Personal Finance Engine.
"""

from dataclasses import dataclass
from typing import List, Dict

# ==========================================
# 1. GLOBAL REPRODUCIBILITY & SIMULATION
# ==========================================

RANDOM_SEED: int = 42
DEFAULT_SIMULATION_DAYS: int = 365 * 2  # 2 years of daily financial history
NUM_SIMULATED_USERS: int = 50          # Cohort size for dataset

# ==========================================
# 2. TRANSACTION CATEGORIES & CLASSIFICATION
# ==========================================

INCOME_CATEGORIES: List[str] = [
    "Salary",
    "Freelance",
    "Investments"
]

ESSENTIAL_EXPENSE_CATEGORIES: List[str] = [
    "Rent",
    "Utilites",
    "Groceries",
    "medical",
    "Transporatation"
]

DISCRETIONARY_EXPENSE_CATEGORIES: List[str] = [
    "Dining_Out",
    "Shopping",
    "Entertainment",
    "Subscriptions",
    "Travel",
]

ALL_EXPENSE_CATEGORIES: List[str] = (
    ESSENTIAL_EXPENSE_CATEGORIES + DISCRETIONARY_EXPENSE_CATEGORIES
)


# ==========================================
# 3. FINANCIAL PROFILES (PERSONAS)
# ==========================================
 
@dataclass(frozen=True)  #Keeps configuration objects immutable so no function can accidentally modify baseline ratios at runtime.
class UserProfileConfig:
    profile_name: str
    base_monthly_income:float
    income_volatility: float       # std dev as % of income
    essential_expense_ratio: float # % of income for fixed needs
    discretionary_mean_ratio: float# % of income for lifestyle
    discretionary_volatility: float# spending fluctuation
    emergency_prob_per_month: float# probability of sudden medical/repair shock
    initial_savings_multiplier: float # initial bank balance as multiple of monthly income
    

USER_PROFILES: Dict[str, UserProfileConfig] = {
    "conservative_saver": UserProfileConfig(
        profile_name="Conservative Saver",
        base_monthly_income=90_000.0,
        income_volatility=0.02,
        essential_expense_ratio=0.35,
        discretionary_mean_ratio=0.15,
        discretionary_volatility=0.08,
        emergency_prob_per_month=0.05,
        initial_savings_multiplier=6.0,  # 6 months emergency fund
    ),
    "moderate_balanced": UserProfileConfig(
        profile_name="Moderate Balanced",
        base_monthly_income=65_000.0,
        income_volatility=0.05,
        essential_expense_ratio=0.45,
        discretionary_mean_ratio=0.30,
        discretionary_volatility=0.18,
        emergency_prob_per_month=0.10,
        initial_savings_multiplier=2.5,  # 2.5 months buffer
    ),
    "high_earner_spender": UserProfileConfig(
        profile_name="High Earner Heavy Spender",
        base_monthly_income=180_000.0,
        income_volatility=0.10,
        essential_expense_ratio=0.30,
        discretionary_mean_ratio=0.55,
        discretionary_volatility=0.40,
        emergency_prob_per_month=0.15,
        initial_savings_multiplier=1.2,  # high income but thin liquid buffer
    ),
    "paycheck_to_paycheck": UserProfileConfig(
        profile_name="Paycheck to Paycheck",
        base_monthly_income=35_000.0,
        income_volatility=0.15,
        essential_expense_ratio=0.65,
        discretionary_mean_ratio=0.30,
        discretionary_volatility=0.25,
        emergency_prob_per_month=0.20,
        initial_savings_multiplier=0.3,  # barely 10 days of runway
    ),
}

# ==========================================
# 4. RISK TIERS & THRESHOLDS (TARGET LABELS)
# ==========================================
# Runway (days of living expenses remaining if income stops)
# Savings Rate = (Income - Expense) / Income
# Risk Score: 0 = Low Risk, 1 = Medium Risk, 2 = High Risk (Crush Vulnerability)
class RiskTier:
    LOW = 0
    MEDIUM = 1
    HIGH = 2
RISK_TIER_LABELS: Dict[int, str] = {
    RiskTier.LOW: "Low Risk (Healthy)",
    RiskTier.MEDIUM: "Medium Risk (Moderate Vulnerability)",
    RiskTier.HIGH: "High Risk (Cash-Flow Distress)",
}