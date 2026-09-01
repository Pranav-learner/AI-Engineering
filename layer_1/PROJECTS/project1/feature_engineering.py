"""
Raw Daily Transactions (Timestamped stream)
              │
              ▼
   Aggregation by Month (t)
              │
              ▼
┌─────────────────────────────────────────────────────────┐
│ 1. Income & Savings Dynamics                            │
│    - Net Savings = Income_t - Expense_t                 │
│    - Savings Rate = Net Savings / Income_t              │
│                                                         │
│ 2. Autoregressive Lag Features                          │
│    - Expense Lag 1 (E_{t-1}), Expense Lag 2 (E_{t-2})   │
│    - Month-over-Month Spend Acceleration                │
│                                                         │
│ 3. Rolling Statistical Volatility                       │
│    - 3-Month Moving Average (MA_3)                      │
│    - 3-Month Rolling Volatility (CV_3)                  │
│    - Outlier Shock Count (MAD anomalies in month)       │
│                                                         │
│ 4. Solvency & Liquidity (Runway)                        │
│    - Daily Burn Rate = Expense_t / 30                   │
│    - Cash Runway (Days) = Balance_t / Daily Burn Rate   │
│                                                         │
│ 5. Spending Entropy (Dispersion)                        │
│    - Essential vs Discretionary Ratio                   │
│    - Shannon Category Entropy: H = -sum(p_i * log(p_i)) │
└─────────────────────────────────────────────────────────┘
              │
              ▼
┌───────────────────────────────┬───────────────────────────────┐
│ Target 1 (Regression):        │ Target 2 (Classification):    │
│ Next Month Expense (E_{t+1})  │ Next Month Risk Tier (0/1/2)  │
└───────────────────────────────┴───────────────────────────────┘
"""
"""
Preventing Data Leakage (Temporal Splitting)
In standard ML (like image classification), random shuffling (train_test_split(shuffle=True)) is fine.

In Time-Series and Financial ML, random shuffling is a catastrophic mistake. If you randomly shuffle months, your model will use Month 10 to predict Month 9, resulting in artificially inflated 99% accuracy that fails completely in production.

We enforce strict Temporal Train / Validation / Test Splitting:

First 70% of chronological months $\to$ Train
Middle 15% of chronological months $\to$ Validation
Final 15% of chronological months $\to$ Test (Unseen Future)
"""


from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from .config import (
    ESSENTIAL_EXPENSE_CATEGORIES,
    RiskTier,
)


from .math_stats_engine import (
    detect_outliers_modified_zscore,
)

def extract_monthly_user_features(df_user_transactions: pd.DataFrame)-> pd.DataFrame:
    """
    Transforms a single user's daily transaction stream into monthly tabular ML features.
    Computes:
      - Total Monthly Income & Expenses
      - Autoregressive Lags (Lag-1, Lag-2)
      - Rolling 3-month statistics (Mean, Std, Volatility CV)
      - Liquidity (Cash Runway in days, Burn Rate)
      - Category dispersion (Essential vs Discretionary Ratio, Shannon Entropy)
      - Regression Target: Next Month's Expense (E_{t+1})
      - Classification Target: Next Month's Risk Tier (0 = Low, 1 = Medium, 2 = High)
    """

    df = df_user_transactions.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["year_month"] = df["date"].dt.to_period("M")

    months = sorted(df["year_month"].unique())
    monthly_rows : List[Dict] = []

    for ym in months:
        m_df = df[df["year_month"] == ym]     

        #1 Monthly income & Expenses
        income_txs = m_df[m_df["transaction_type"] == "INCOME"]
        expense_txs = m_df[m_df["transaction_type"] == "EXPENSE"]

        monthly_income = float(income_txs["amount"].sum())
        monthly_expense = float(expense_txs["amount"].sum())   
        ending_balance = float(m_df.iloc[-1]["balance_after"]) if len(m_df) > 0 else 0.0

        #2 Savings Dynamics
        net_savings = monthly_income - monthly_expense
        savings_rate = (net_savings / monthly_income) if monthly_income > 0 else -1.0
        savings_rate = float(np.clip(savings_rate, -2.0, 1.0))

        # 3. Solvency & Liquidity (Burn rate & Runway)
        daily_burn_rate = max(1.0, monthly_expense / 30.0)
        runway_days = max(0.0, ending_balance / daily_burn_rate)

        # 4. Essential vs Discretionary Ratio
        essential_spend = float(expense_txs[expense_txs["category"].isin(ESSENTIAL_EXPENSE_CATEGORIES)]["amount"].sum())
        discretionary_spend = max(0.0, monthly_expense - essential_spend)
        essential_ratio = (essential_spend / monthly_expense) if monthly_expense > 0 else 0.5
        
        # 5. Shannon Entropy of Category Spending (H = -sum(p * log(p)))
        cat_group = expense_txs.groupby("category")["amount"].sum()
        if monthly_expense > 0 and len(cat_group) > 0:
            probs = (cat_group / monthly_expense).values
            probs = probs[probs > 0]
            spending_entropy = float(-np.sum(probs * np.log2(probs)))
        else:
            spending_entropy = 0.0
        # 6. Outlier / Emergency Spikes in Month (Using MAD)
        expense_amounts = expense_txs["amount"].values
        outlier_mask = detect_outliers_modified_zscore(expense_amounts, threshold=3.5)
        num_outliers = int(np.sum(outlier_mask))
        monthly_rows.append({
            "user_id": df["user_id"].iloc[0],
            "user_profile": df["user_profile"].iloc[0],
            "year_month": str(ym),
            "monthly_income": round(monthly_income, 2),
            "monthly_expense": round(monthly_expense, 2),
            "ending_balance": round(ending_balance, 2),
            "net_savings": round(net_savings, 2),
            "savings_rate": round(savings_rate, 4),
            "daily_burn_rate": round(daily_burn_rate, 2),
            "runway_days": round(runway_days, 1),
            "essential_ratio": round(essential_ratio, 4),
            "discretionary_spend": round(discretionary_spend, 2),
            "spending_entropy": round(spending_entropy, 4),
            "num_outliers": num_outliers,
        })
    feat_df = pd.DataFrame(monthly_rows)


# -------------------------------------------------------------
    # 7. TIME-SERIES LAGS & ROLLING WINDOW FEATURES
    # -------------------------------------------------------------
    feat_df["expense_lag_1"] = feat_df["monthly_expense"].shift(1)
    feat_df["expense_lag_2"] = feat_df["monthly_expense"].shift(2)
    feat_df["income_lag_1"] = feat_df["monthly_income"].shift(1)
    # Rolling 3-Month Statistics
    feat_df["rolling_expense_mean_3m"] = feat_df["monthly_expense"].rolling(window=3).mean()
    feat_df["rolling_expense_std_3m"] = feat_df["monthly_expense"].rolling(window=3).std(ddof=1).fillna(0.0)
    feat_df["rolling_expense_cv_3m"] = (
        feat_df["rolling_expense_std_3m"] / feat_df["rolling_expense_mean_3m"]
    ).fillna(0.0)
    # Month-over-Month Spending Acceleration
    feat_df["spend_mom_change"] = (
        (feat_df["monthly_expense"] - feat_df["expense_lag_1"]) / (feat_df["expense_lag_1"] + 1e-5)
    ).fillna(0.0)


    # ------------------------------------------------------------
    # 8. DEFINE TARGET LABELS (Next Month t+1)
    # -------------------------------------------------------------
    # Target 1 (Regression): Next Month's Expense E_{t+1}
    feat_df["target_next_month_expense"] = feat_df["monthly_expense"].shift(-1)

    # Target 2 (Classification): Next Month's Risk Tier
    next_runway = feat_df["runway_days"].shift(-1)
    next_savings_rate = feat_df["savings_rate"].shift(-1)

    def assign_risk_tier(runway: float, s_rate: float) -> int:
        if pd.isna(runway) or pd.isna(s_rate):
            return np.nan
        # High Risk: Less than 30 days runway OR negative savings rate
        if runway < 30.0 or s_rate < 0.0:
            return RiskTier.HIGH
        # Medium Risk: 30 to 90 days runway OR thin savings buffer (< 15%)
        elif runway < 90.0 or s_rate < 0.15:
            return RiskTier.MEDIUM
        # Low Risk: > 90 days runway AND healthy savings rate >= 15%
        else:
            return RiskTier.LOW

    feat_df["target_risk_tier"] = [
        assign_risk_tier(r, s) for r, s in zip(next_runway, next_savings_rate)
    ]

    # Drop the first 2 rows (due to lag_2 NaN) and the last row (due to shift(-1) target NaN)
    clean_feat_df = feat_df.dropna().reset_index(drop=True)
    clean_feat_df["target_risk_tier"] = clean_feat_df["target_risk_tier"].astype(int)
    return clean_feat_df


def build_cohort_feature_matrix(df_cohort_transactions: pd.DataFrame) -> pd.DataFrame:
    """
    Builds the full multi-user feature matrix across all users in the cohort.
    """
    user_ids = df_cohort_transactions["user_id"].unique()
    all_user_matrices: List[pd.DataFrame] = []
    for uid in user_ids:
        user_tx = df_cohort_transactions[df_cohort_transactions["user_id"] == uid]
        user_feat = extract_monthly_user_features(user_tx)
        all_user_matrices.append(user_feat)
    full_matrix = pd.concat(all_user_matrices, ignore_index=True)
    return full_matrix
FEATURE_COLUMNS: List[str] = [
    "monthly_income",
    "monthly_expense",
    "ending_balance",
    "net_savings",
    "savings_rate",
    "daily_burn_rate",
    "runway_days",
    "essential_ratio",
    "discretionary_spend",
    "spending_entropy",
    "num_outliers",
    "expense_lag_1",
    "expense_lag_2",
    "income_lag_1",
    "rolling_expense_mean_3m",
    "rolling_expense_std_3m",
    "rolling_expense_cv_3m",
    "spend_mom_change",
]
def temporal_train_val_test_split(
    df_features: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Performs strict chronological time-series splitting to guarantee ZERO future data leakage.
    Train: First 70% months -> Validation: Next 15% months -> Test: Final 15% months.
    """
    all_months = sorted(df_features["year_month"].unique())
    n_months = len(all_months)
    train_end_idx = int(n_months * train_ratio)
    val_end_idx = int(n_months * (train_ratio + val_ratio))
    train_months = all_months[:train_end_idx]
    val_months = all_months[train_end_idx:val_end_idx]
    test_months = all_months[val_end_idx:]
    train_df = df_features[df_features["year_month"].isin(train_months)].reset_index(drop=True)
    val_df = df_features[df_features["year_month"].isin(val_months)].reset_index(drop=True)
    test_df = df_features[df_features["year_month"].isin(test_months)].reset_index(drop=True)
    return train_df, val_df, test_df
if __name__ == "__main__":
    from .data_generator import generate_cohort_dataset
    print("Generating raw transaction stream for feature engineering...")
    tx_df = generate_cohort_dataset(num_users=8, num_days=730)
    print(f"Generated {len(tx_df):,} transactions.")
    print("\nExtracting tabular feature matrices...")
    feat_matrix = build_cohort_feature_matrix(tx_df)
    print(f"Constructed feature matrix: {feat_matrix.shape[0]} monthly records x {feat_matrix.shape[1]} columns.")
    print("\n--- Sample Feature Matrix (Top 5 rows) ---")
    display_cols = ["user_id", "year_month", "monthly_income", "monthly_expense", "runway_days", "target_next_month_expense", "target_risk_tier"]
    print(feat_matrix[display_cols].head())
    print("\n--- Target Risk Tier Distribution ---")
    print(feat_matrix["target_risk_tier"].value_counts().rename(index={0: "0 (Low)", 1: "1 (Medium)", 2: "2 (High)"}))
    train_df, val_df, test_df = temporal_train_val_test_split(feat_matrix)
    print(f"\nTemporal Split: Train={len(train_df)} rows, Val={len(val_df)} rows, Test={len(test_df)} rows.")
