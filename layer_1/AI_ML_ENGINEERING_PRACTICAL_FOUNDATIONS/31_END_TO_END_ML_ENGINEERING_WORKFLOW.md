# 🏆 Module 31: End-to-End Production ML Engineering Capstone Workflow

---

## 1. The Complete Integrated Pipeline

This document unites every module into a complete, working, production-grade Python script that can be used as a template for all your future AI projects.

```
Raw Transaction Stream (data_generator.py)
              │
              ▼
Data Cleaning & Statistical Engine (math_stats_engine.py)
              │
              ▼
Feature Engineering & Lags (feature_engineering.py)
              │
              ▼
Leak-Proof Temporal Splitting (70% Train / 15% Val / 15% Test)
              │
              ▼
Scikit-Learn Preprocessor Pipelines (ColumnTransformer, StandardScaler)
              │
              ▼
Model Tournament: Baselines vs. Classical ML vs. PyTorch MLP
              │
              ▼
Adversarial Stress-Testing & Leakage Audit
              │
              ▼
Production Inference & CLI Diagnostic Card (app_cli.py)
```

---

## 2. The Master Python Code Blueprint

```python
"""
production_pipeline_template.py - Reusable End-to-End Machine Learning Pipeline Blueprint.
"""

from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# =====================================================================
# 1. DATA PREPARATION & TEMPORAL SPLITTING
# =====================================================================

def prepare_and_split_data(df: pd.DataFrame, feature_cols: List[str], target_col: str):
    """
    Performs chronological split to ensure zero future data leakage.
    """
    n = len(df)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    train_df = df.iloc[:train_end].reset_index(drop=True)
    val_df = df.iloc[train_end:val_end].reset_index(drop=True)
    test_df = df.iloc[val_end:].reset_index(drop=True)

    X_train, y_train = train_df[feature_cols], train_df[target_col].values
    X_val, y_val = val_df[feature_cols], val_df[target_col].values
    X_test, y_test = test_df[feature_cols], test_df[target_col].values

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


# =====================================================================
# 2. BASELINE ESTIMATOR
# =====================================================================

class RollingMeanBaseline(BaseEstimator, RegressorMixin):
    def fit(self, X, y=None):
        self.fallback_mean_ = float(np.mean(y)) if y is not None else 0.0
        return self

    def predict(self, X):
        if "rolling_mean_3m" in X.columns:
            return X["rolling_mean_3m"].fillna(self.fallback_mean_).values
        return np.full(len(X), self.fallback_mean_)


# =====================================================================
# 3. PYTORCH DEEP LEARNING MODEL
# =====================================================================

class PyTorchTabularModel(BaseEstimator):
    def __init__(self, in_features: int, epochs: int = 100, lr: float = 0.005):
        self.in_features = in_features
        self.epochs = epochs
        self.lr = lr
        self.scaler = StandardScaler()
        self.model = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def fit(self, X, y):
        X_scaled = self.scaler.fit_transform(X)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        criterion = nn.MSELoss()

        dataset = TensorDataset(torch.tensor(X_scaled, dtype=torch.float32), torch.tensor(y, dtype=torch.float32).unsqueeze(1))
        loader = DataLoader(dataset, batch_size=16, shuffle=True)

        self.model.train()
        for epoch in range(self.epochs):
            for batch_x, batch_y in loader:
                optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
        return self

    def predict(self, X):
        self.model.eval()
        X_scaled = self.scaler.transform(X)
        with torch.no_grad():
            preds = self.model(torch.tensor(X_scaled, dtype=torch.float32)).squeeze(1).numpy()
        return preds


# =====================================================================
# 4. TOURNAMENT EVALUATION RUNNER
# =====================================================================

def run_production_tournament(X_train, y_train, X_test, y_test, in_features: int):
    models = {
        "1. Baseline (3m Mean)": RollingMeanBaseline(),
        "2. Ridge Regression": Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=10.0))]),
        "3. Random Forest": Pipeline([("scaler", StandardScaler()), ("model", RandomForestRegressor(n_estimators=100, max_depth=6))]),
        "4. PyTorch Tabular MLP": PyTorchTabularModel(in_features=in_features, epochs=100),
    }

    print("=" * 60)
    print("🏆 MODEL TOURNAMENT BENCHMARK RESULTS")
    print("=" * 60)
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)
        print(f"{name:<25}: MAE = ₹{mae:,.2f} | R² = {r2:.4f}")
```

---

## 3. The AI Engineer's Daily Mindset

```
Never blindly copy code.
Always ask:
1. What is the shape before?
2. What operation is occurring?
3. What is the shape after?
4. Is there any future data leakage?
5. Does this model beat the simple baseline?
```
