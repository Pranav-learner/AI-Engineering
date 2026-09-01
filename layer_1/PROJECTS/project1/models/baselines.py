"""
Why Every AI Engineer Must Start with Baselines:
In real-world AI engineering, you should NEVER train a complex Machine Learning or Deep Learning model without comparing it against a simple baseline.

If a complex 50,000-parameter Neural Network achieves $R^2 = 0.82$, but a 3-line Rolling Mean achieves $R^2 = 0.80$, the neural network does not justify its complexity, latency, and maintenance cost.
A baseline gives us the minimum performance threshold that any ML model must convincingly beat.

1. If a complex 50,000-parameter Neural Network achieves $R^2 = 0.82$, but a 3-line Rolling Mean achieves $R^2 = 0.80$, the neural network does not justify its complexity, latency, and maintenance cost.
2. A baseline gives us the minimum performance threshold that any ML model must convincingly beat.
"""

from typing import Optional
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from ..config import RiskTier

class HistoricalMeanForecaster(BaseEstimator, RegressorMixin):
    """
    Baseline Forecaster:
    Predicts next month's expense as simply the user's recent rolling average expense.
    Uses 'rolling_expense_mean_3m' or 'monthly_expense' if available, otherwise global train mean.
    """

    def __init__(self, fallback_mean: float = 0.0):
        self.fallback_mean = fallback_mean

    def fit(self, X: pd.DataFrame, y: Optional[np.ndarray] = None):
        if y is not None and len(y) > 0:
            self.fallback_mean = float(np.mean(y))
        elif "monthly_expense" in X.columns:
            self.fallback_mean = float(X["monthly_expense"] .mean())
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if "rolling_expense_mean_3m" in X.columns:
            preds = X["rolling_expense_mean_3m"].values.copy()
            # If any NaN, fill with fallback
            preds = np.where(np.isnan(preds), self.fallback_mean, preds)
            return preds
        elif "monthly_expense" in X.columns:
            return X["monthly_expense"].values
        else:
            return np.full(len(X), self.fallback_mean)

    
class RuleBasedRiskClassifier(BaseEstimator, ClassifierMixin):
    """
    Baseline Heuristic Classifier:
    Uses deterministic domain heuristics:
      - If runway < 30 days OR savings_rate < 0 -> High Risk (2)
      - Else if runway < 90 days OR savings_rate < 0.15 -> Medium Risk (1)
      - Else -> Low Risk (0)
    """
    def fit(self, X: pd.DataFrame, y: Optional[np.ndarray] = None):
        # Deterministic rule: No training parameters to fit
        return self
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        runway = X["runway_days"].values if "runway_days" in X.columns else np.zeros(len(X))
        savings_rate = X["savings_rate"].values if "savings_rate" in X.columns else np.zeros(len(X))
        preds = np.zeros(len(X), dtype=int)
        for i in range(len(X)):
            r = runway[i]
            s = savings_rate[i]
            if r < 30.0 or s < 0.0:
                preds[i] = RiskTier.HIGH
            elif r < 90.0 or s < 0.15:
                preds[i] = RiskTier.MEDIUM
            else:
                preds[i] = RiskTier.LOW
        return preds