"""
app_cli.py - Interactive Terminal Dashboard for the Personal Finance Risk Engine.
"""

import sys
from typing import Dict
import numpy as np
import pandas as pd
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import RANDOM_SEED, RISK_TIER_LABELS, RiskTier, USER_PROFILES
from .data_generator import generate_cohort_dataset, generate_single_user_transactions
from .feature_engineering import (
    FEATURE_COLUMNS,
    build_cohort_feature_matrix,
    extract_monthly_user_features,
    temporal_train_val_test_split,
)
from .math_stats_engine import (
    calculate_expense_confidence_interval,
    calculate_volatility_metrics,
    detect_outliers_modified_zscore,
)

console = Console()


# =====================================================================
# 1. TRAIN PRODUCTION PRODUCTION MODELS
# =====================================================================

def train_production_models() -> Tuple_Models := tuple:
    """
    Trains production Random Forest Regressor and Classifier on historical cohort data.
    """
    with console.status("[bold green]Training Production ML & Risk Engines on Historical Cohort..."):
        tx_df = generate_cohort_dataset(num_users=32, num_days=730)
        feat_df = build_cohort_feature_matrix(tx_df)
        train_df, _, _ = temporal_train_val_test_split(feat_df, train_ratio=0.85, val_ratio=0.15)

        X_train = train_df[FEATURE_COLUMNS]
        y_reg = train_df["target_next_month_expense"].values
        y_cls = train_df["target_risk_tier"].values

        reg_pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("model", RandomForestRegressor(n_estimators=100, max_depth=6, random_state=RANDOM_SEED)),
        ])
        reg_pipeline.fit(X_train, y_reg)

        cls_pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("model", RandomForestClassifier(n_estimators=100, max_depth=6, class_weight="balanced", random_state=RANDOM_SEED)),
        ])
        cls_pipeline.fit(X_train, y_cls)

    return reg_pipeline, cls_pipeline


# =====================================================================
# 2. GENERATE USER DIAGNOSTIC CARD
# =====================================================================

def analyze_and_display_user(
    user_id: str,
    profile_key: str,
    reg_model: Pipeline,
    cls_model: Pipeline,
):
    """
    Simulates a user, extracts their stats & features, runs ML inference, and displays a rich card.
    """
    # 1. Generate user history
    user_tx = generate_single_user_transactions(user_id=user_id, profile_key=profile_key, num_days=365)
    feat_df = extract_monthly_user_features(user_tx)
    
    if len(feat_df) == 0:
        console.print(f"[red]Error: Not enough history for user {user_id}[/red]")
        return

    latest_snapshot = feat_df.iloc[-1]
    latest_features = feat_df[FEATURE_COLUMNS].iloc[[-1]]

    # 2. Math & Stats Engine calculations
    expense_txs = user_tx[user_tx["transaction_type"] == "EXPENSE"]
    all_expenses = expense_txs["amount"].values
    monthly_spends = feat_df["monthly_expense"].values

    volatility = calculate_volatility_metrics(all_expenses)
    outliers = detect_outliers_modified_zscore(all_expenses, threshold=3.5)
    num_outliers = int(np.sum(outliers))
    ci = calculate_expense_confidence_interval(monthly_spends, confidence_level=0.95)

    # 3. ML Inference
    pred_expense = float(reg_model.predict(latest_features)[0])
    pred_risk = int(cls_model.predict(latest_features)[0])
    pred_risk_proba = cls_model.predict_proba(latest_features)[0]

    # 4. Format Rich Dashboard
    profile_name = latest_snapshot["user_profile"]
    balance = latest_snapshot["ending_balance"]
    income = latest_snapshot["monthly_income"]
    expense = latest_snapshot["monthly_expense"]
    runway = latest_snapshot["runway_days"]
    savings_rate = latest_snapshot["savings_rate"] * 100.0

    # Risk Styling
    if pred_risk == RiskTier.LOW:
        risk_badge = "[bold green]🟢 LOW RISK (STABLE & HEALTHY)[/bold green]"
        recommendation = "✅ Maintain current savings allocation. Ready for long-term investments."
    elif pred_risk == RiskTier.MEDIUM:
        risk_badge = "[bold yellow]🟡 MEDIUM RISK (MODERATE VULNERABILITY)[/bold yellow]"
        recommendation = "⚠️ Build 2-3 months of additional cash buffer. Reduce discretionary weekend spends."
    else:
        risk_badge = "[bold red]🔴 HIGH RISK (CASH-FLOW CRUNCH VULNERABILITY)[/bold red]"
        recommendation = "🚨 URGENT: Runway < 30 days! Immediately freeze discretionary purchases and replenish reserve fund."

    # Build Summary Table
    table = Table(show_header=True, header_style="bold cyan", expand=True)
    table.add_column("Metric", style="bold white", width=28)
    table.add_column("Observed Value", justify="right")
    table.add_column("Statistical / ML Diagnosis", justify="left")

    table.add_row("Current Bank Balance", f"₹{balance:,.2f}", "Liquid Available Capital")
    table.add_row("Monthly Income", f"₹{income:,.2f}", "Latest Inflow")
    table.add_row("Monthly Living Expenses", f"₹{expense:,.2f}", f"Savings Rate: {savings_rate:+.1f}%")
    table.add_row("Cash Runway", f"{runway:.1f} Days", "Days to depletion if income stops")
    table.add_row("Spending Volatility (CV)", f"{volatility['cv']:.3f}", "Scale-independent dispersion" if volatility['cv'] < 0.25 else "[yellow]High spending swings[/yellow]")
    table.add_row("Emergency Shock Spikes", f"{num_outliers} events", "Detected via Robust MAD Outlier Engine")
    table.add_row("95% Expense Confidence Interval", f"₹{ci['ci_lower']:,.0f} - ₹{ci['ci_upper']:,.0f}", f"Expected Monthly Base: ₹{ci['mean']:,.0f}")
    table.add_row("ML Forecast Next Month Expense", f"[bold cyan]₹{pred_expense:,.2f}[/bold cyan]", "Random Forest Regressor Inference")
    table.add_row("ML Cash-Flow Risk Tier", risk_badge, f"Prob: Low={pred_risk_proba[0]:.2f}, Med={pred_risk_proba[1]:.2f}, High={pred_risk_proba[2]:.2f}")

    panel_content = Text()
    panel_title = f"💳 PERSONAL FINANCE INTELLIGENCE REPORT — [{user_id}] {profile_name.upper()}"
    
    console.print("\n")
    console.print(Panel(table, title=panel_title, subtitle=f"[italic white]{recommendation}[/italic white]", expand=True, border_style="bright_blue"))


# =====================================================================
# MAIN RUNNER
# =====================================================================

def main():
    console.print(Panel.fit(
        "[bold green]Personal Finance Risk & Cash-Flow Intelligence Engine[/bold green]\n"
        "[italic white]Mathematics + Statistics + Machine Learning + Deep Learning System[/italic white]",
        border_style="green",
    ))

    # Train Models
    reg_model, cls_model = train_production_models()
    console.print("[bold green]✓ Production Models Loaded Successfully![/bold green]\n")

    # Run Analysis across all 4 Financial Personas
    personas = [
        ("USR_001", "conservative_saver"),
        ("USR_002", "moderate_balanced"),
        ("USR_003", "high_earner_spender"),
        ("USR_004", "paycheck_to_paycheck"),
    ]

    for uid, p_key in personas:
        analyze_and_display_user(uid, p_key, reg_model, cls_model)

    console.print("\n[bold green]🎉 Project 1 Full Diagnostic Execution Completed Successfully![/bold green]\n")


if __name__ == "__main__":
    main()
