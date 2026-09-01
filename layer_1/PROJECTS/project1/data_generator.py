"""
data_generator.py - Generates realistic multi-profile personal banking transaction streams.
"""

from datetime import datetime
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import numpy as np 
import pandas as pd 

from .config import (
    ALL_EXPENSE_CATEGORIES,
    DISCRETIONARY_EXPENSE_CATEGORIES,
    INCOME_CATEGORIES,
    RANDOM_SEED,
    USER_PROFILES,
    UserProfileConfig,
)

def generate_single_user_transactions(
    user_id: str,
    profile_key: str,
    num_days: int = 730,
    start_date: datetime = datetime(2024,1,1),
    rng: Optional[np.random.Generator] = None,
) -> pd.DataFrame:
    """
    Generate a full time-series of banking transactions for a single user.
    Simulates:
      -Monthly salary and freelance spikes 
      - Fixed recurring expenses (Rent, Utilities)
      - Daily/Weekly variable living expenses with weekend seasonality
      - Rare high-impact financial shocks
      - Daily continuous balance tracking
    """

    if rng is None:
        rng = np.random.default_rng(RANDOM_SEED)

    profile: UserProfileConfig = USER_PROFILES[profile_key]
     # Add individual variation to base income (+/- 10%)

    individual_income = profile.base_monthly_income * rng.uniform(0.90,1.10)
    starting_balance = individual_income * profile.initial_savings_multiplier
    current_balance = starting_balance

    records: List[Dict] = []
    tx_counter = 0

    for day_offset in range(num_days):
        current_date = start_date + timedelta(days=day_offset)
        day_of_month = current_date.day
        is_weekend = (current_date.weekday() >= 5)

        # -------------------------------------------------------------
        # 1. INCOME STREAM (Salary around 1st of each month)
        # -------------------------------------------------------------
        if day_of_month == 1:
            salary_variation = rng.normal(1.0, profile.income_volatility)
            salary_amount = round(individual_income * max(0.7, salary_variation), 2)
            current_balance += salary_amount
            tx_counter += 1
            records.append({
                "transaction_id": f"TX_{user_id}_{tx_counter:06d}",
                "user_id": user_id,
                "user_profile": profile.profile_name,
                "date": current_date,
                "category": "Salary",
                "transaction_type": "INCOME",
                "amount": salary_amount,
                "balance_after": round(current_balance, 2),
            })

            # Occasional freelance / bonus income (5% chance on any day)
        if rng.random() < 0.05:
            freelance_amount = round(float(rng.lognormal(mean=7.5, sigma=0.6)), 2)
            current_balance += freelance_amount
            tx_counter += 1
            records.append({
                "transaction_id": f"TX_{user_id}_{tx_counter:06d}",
                "user_id": user_id,
                "user_profile": profile.profile_name,
                "date": current_date,
                "category": "Freelance",
                "transaction_type": "INCOME",
                "amount": freelance_amount,
                "balance_after": round(current_balance, 2),
            })

            # -------------------------------------------------------------
        # 2. FIXED ESSENTIAL EXPENSES (Rent & Utilities)
        # -------------------------------------------------------------
        # Rent on 5th of the month
        if day_of_month == 5:
            monthly_essential_budget = individual_income * profile.essential_expense_ratio
            rent_amount = round(monthly_essential_budget * 0.60 * rng.uniform(0.98, 1.02), 2)
            current_balance -= rent_amount
            tx_counter += 1
            records.append({
                "transaction_id": f"TX_{user_id}_{tx_counter:06d}",
                "user_id": user_id,
                "user_profile": profile.profile_name,
                "date": current_date,
                "category": "Rent",
                "transaction_type": "EXPENSE",
                "amount": rent_amount,
                "balance_after": round(current_balance, 2),
            })
        # Utilities on 10th of the month
        if day_of_month == 10:
            monthly_essential_budget = individual_income * profile.essential_expense_ratio
            utility_amount = round(monthly_essential_budget * 0.15 * rng.uniform(0.85, 1.20), 2)
            current_balance -= utility_amount
            tx_counter += 1
            records.append({
                "transaction_id": f"TX_{user_id}_{tx_counter:06d}",
                "user_id": user_id,
                "user_profile": profile.profile_name,
                "date": current_date,
                "category": "Utilities",
                "transaction_type": "EXPENSE",
                "amount": utility_amount,
                "balance_after": round(current_balance, 2),
            })
        # -------------------------------------------------------------
        # 3. DAILY / WEEKLY VARIABLE EXPENSES
        # -------------------------------------------------------------
        # Groceries (every 2-3 days, roughly 40% probability per day)
        if rng.random() < 0.40:
            grocery_base = (individual_income * profile.essential_expense_ratio * 0.25) / 10.0
            grocery_amount = round(float(rng.lognormal(mean=np.log(max(grocery_base, 100)), sigma=0.35)), 2)
            current_balance -= grocery_amount
            tx_counter += 1
            records.append({
                "transaction_id": f"TX_{user_id}_{tx_counter:06d}",
                "user_id": user_id,
                "user_profile": profile.profile_name,
                "date": current_date,
                "category": "Groceries",
                "transaction_type": "EXPENSE",
                "amount": grocery_amount,
                "balance_after": round(current_balance, 2),
            })
        # Discretionary spending (Dining, Entertainment, Shopping)
        # Higher probability and higher amounts on weekends
        num_discretionary_events = rng.poisson(lam=1.5 if is_weekend else 0.7)
        for _ in range(num_discretionary_events):
            chosen_cat = rng.choice(DISCRETIONARY_EXPENSE_CATEGORIES)
            daily_disc_mean = (individual_income * profile.discretionary_mean_ratio) / 30.0
            weekend_boost = 1.5 if is_weekend else 1.0
            disc_amount = round(
                float(rng.lognormal(
                    mean=np.log(max(daily_disc_mean * weekend_boost, 150)),
                    sigma=profile.discretionary_volatility,
                )),
                2,
            )
            current_balance -= disc_amount
            tx_counter += 1
            records.append({
                "transaction_id": f"TX_{user_id}_{tx_counter:06d}",
                "user_id": user_id,
                "user_profile": profile.profile_name,
                "date": current_date,
                "category": chosen_cat,
                "transaction_type": "EXPENSE",
                "amount": disc_amount,
                "balance_after": round(current_balance, 2),
            })
        # -------------------------------------------------------------
        # 4. EMERGENCY FINANCIAL SHOCK (Outlier event)
        # -------------------------------------------------------------
        daily_shock_prob = profile.emergency_prob_per_month / 30.0
        if rng.random() < daily_shock_prob:
            shock_amount = round(individual_income * float(rng.uniform(0.40, 1.20)), 2)
            current_balance -= shock_amount
            tx_counter += 1
            records.append({
                "transaction_id": f"TX_{user_id}_{tx_counter:06d}",
                "user_id": user_id,
                "user_profile": profile.profile_name,
                "date": current_date,
                "category": "Medical",
                "transaction_type": "EXPENSE",
                "amount": shock_amount,
                "balance_after": round(current_balance, 2),
            })
    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])
    return df
def generate_cohort_dataset(
    num_users: int = 50,
    num_days: int = 730,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """
    Generates a combined dataset for a cohort of users evenly distributed
    across the 4 financial personas.
    """
    rng = np.random.default_rng(seed)
    profile_keys = list(USER_PROFILES.keys())
    all_dfs: List[pd.DataFrame] = []
    for i in range(num_users):
        user_id = f"USR_{i+1:03d}"
        chosen_profile = profile_keys[i % len(profile_keys)]
        user_df = generate_single_user_transactions(
            user_id=user_id,
            profile_key=chosen_profile,
            num_days=num_days,
            rng=rng,
        )
        all_dfs.append(user_df)
    combined_df = pd.concat(all_dfs, ignore_index=True)
    return combined_df
if __name__ == "__main__":
    print("Generating sample cohort transactions dataset...")
    df = generate_cohort_dataset(num_users=4, num_days=180)
    print(f"Generated {len(df):,} transactions across {df['user_id'].nunique()} users.")
    print("\n--- Sample Transactions ---")
    print(df.head(10)[["transaction_id", "user_id", "date", "category", "transaction_type", "amount", "balance_after"]])
    print("\n--- Breakdown by Profile ---")
    print(df.groupby("user_profile")["amount"].agg(["count", "mean", "std", "max"]))
