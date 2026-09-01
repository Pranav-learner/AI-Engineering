"""
classical_models.py - Trains and evaluates Scikit-Learn regression and classification pipelines.
"""

from typing import Dict, Tuple
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_absolute_percentage_error,
    precision_score,
    r2_score,
    recall_score,
    root_mean_squared_error,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ..config import RANDOM_SEED, RISK_TIER_LABELS
from ..feature_engineering import FEATURE_COLUMNS


# =====================================================================
# 1. REGRESSION EVALUATOR & MODEL FACTORIES
# =====================================================================

def evaluate_regression(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Computes standard regression metrics: MAE, RMSE, R^2, and MAPE.
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    return {
        "MAE": round(float(mae), 2),
        "RMSE": round(float(rmse), 2),
        "R2": round(float(r2), 4),
        "MAPE": round(float(mape * 100), 2),  # in %
    }


def get_classical_regressors() -> Dict[str, Pipeline]:
    """
    Returns dictionary of named classical regression pipelines with StandardScaler.
    """
    return {
        "Linear Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LinearRegression()),
        ]),
        "Ridge Regression (L2)": Pipeline([
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=10.0, random_state=RANDOM_SEED)),
        ]),
        "Random Forest Regressor": Pipeline([
            ("scaler", StandardScaler()),
            ("model", RandomForestRegressor(n_estimators=100, max_depth=6, random_state=RANDOM_SEED)),
        ]),
    }


# =====================================================================
# 2. CLASSIFICATION EVALUATOR & MODEL FACTORIES
# =====================================================================

def evaluate_classification(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, object]:
    """
    Computes multi-class classification metrics: Accuracy, Balanced Acc, Precision, Recall, Macro F1, and Confusion Matrix.
    """
    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    return {
        "Accuracy": round(float(acc), 4),
        "Balanced_Accuracy": round(float(bal_acc), 4),
        "Precision_Macro": round(float(prec), 4),
        "Recall_Macro": round(float(rec), 4),
        "F1_Macro": round(float(f1), 4),
        "Confusion_Matrix": cm,
    }


def get_classical_classifiers() -> Dict[str, Pipeline]:
    """
    Returns dictionary of named classical classification pipelines.
    """
    return {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=RANDOM_SEED)),
        ]),
        "Random Forest Classifier": Pipeline([
            ("scaler", StandardScaler()),
            ("model", RandomForestClassifier(
                n_estimators=100, max_depth=6, class_weight="balanced", random_state=RANDOM_SEED
            )),
        ]),
    }


if __name__ == "__main__":
    from ..data_generator import generate_cohort_dataset
    from ..feature_engineering import build_cohort_feature_matrix, temporal_train_val_test_split
    from .baselines import HistoricalMeanForecaster, RuleBasedRiskClassifier

    print("Generating and preparing data...")
    tx_df = generate_cohort_dataset(num_users=16, num_days=730)
    feat_df = build_cohort_feature_matrix(tx_df)
    train_df, val_df, test_df = temporal_train_val_test_split(feat_df)

    X_train, y_reg_train, y_cls_train = train_df[FEATURE_COLUMNS], train_df["target_next_month_expense"].values, train_df["target_risk_tier"].values
    X_test, y_reg_test, y_cls_test = test_df[FEATURE_COLUMNS], test_df["target_next_month_expense"].values, test_df["target_risk_tier"].values

    print(f"\nTraining on {len(X_train)} samples, testing on {len(X_test)} samples.\n")

    # 1. Evaluate Regression Track
    print("=" * 60)
    print("📈 TRACK 1: EXPENSE FORECASTING (REGRESSION)")
    print("=" * 60)

    # Baseline
    mean_baseline = HistoricalMeanForecaster()
    mean_baseline.fit(X_train, y_reg_train)
    base_preds = mean_baseline.predict(X_test)
    print(f"Historical Mean Baseline : {evaluate_regression(y_reg_test, base_preds)}")

    # ML Models
    for name, pipeline in get_classical_regressors().items():
        pipeline.fit(X_train, y_reg_train)
        preds = pipeline.predict(X_test)
        metrics = evaluate_regression(y_reg_test, preds)
        print(f"{name:<25}: {metrics}")

    # 2. Evaluate Classification Track
    print("\n" + "=" * 60)
    print("🎯 TRACK 2: CASH-FLOW RISK CLASSIFICATION")
    print("=" * 60)

    # Baseline
    rule_baseline = RuleBasedRiskClassifier()
    rule_baseline.fit(X_train, y_cls_train)
    rule_preds = rule_baseline.predict(X_test)
    base_cls_metrics = evaluate_classification(y_cls_test, rule_preds)
    print(f"Rule Heuristic Baseline : F1={base_cls_metrics['F1_Macro']}, Acc={base_cls_metrics['Accuracy']}")

    # ML Models
    for name, pipeline in get_classical_classifiers().items():
        pipeline.fit(X_train, y_cls_train)
        preds = pipeline.predict(X_test)
        metrics = evaluate_classification(y_cls_test, preds)
        print(f"{name:<25}: F1={metrics['F1_Macro']}, Acc={metrics['Accuracy']}, Balanced_Acc={metrics['Balanced_Accuracy']}")
        print(f"   Confusion Matrix:\n{metrics['Confusion_Matrix']}")
