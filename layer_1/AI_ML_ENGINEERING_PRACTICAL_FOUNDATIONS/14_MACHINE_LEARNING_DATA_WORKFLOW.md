# 🔄 Module 14: The End-to-End Machine Learning Workflow

---

## 1. The 8 Stages of Production ML Engineering

```
RAW CSV / SQL DATABASE
         │
         ▼
1. INGESTION & DATA CLEANING (Handling NaNs, datatypes, deduplication)
         │
         ▼
2. FEATURE ENGINEERING (Lags, rolling stats, ratios, domain signals)
         │
         ▼
3. X and y EXTRACTION (Feature Matrix X ∈ ℝ^(N x d), Target Vector y ∈ ℝ^N)
         │
         ▼
4. LEAK-PROOF SPLITTING (Strict temporal train / val / test partitions)
         │
         ▼
5. PREPROCESSING PIPELINE (Fitted ONLY on X_train; transforms X_test)
         │
         ▼
6. BASELINE BENCHMARK (Mean forecaster & deterministic heuristic rules)
         │
         ▼
7. MODEL TOURNAMENT (Linear, Ridge, Lasso, Random Forest, PyTorch MLP)
         │
         ▼
8. STRESS-TESTING & DIAGNOSTICS (Adversarial shocks, leakage audit, trade-off analysis)
```

---

## 2. Code Template for the Entire Workflow

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# 1. Feature matrix X and target y
FEATURE_COLS = ["monthly_income", "expense_lag_1", "runway_days", "savings_rate"]
X = df[FEATURE_COLS]
y = df["target_next_month_expense"].values

# 2. Strict splitting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, shuffle=False)

print(f"X_train Shape: {X_train.shape} | y_train Shape: {y_train.shape}")
print(f"X_test Shape : {X_test.shape}  | y_test Shape : {y_test.shape}")
```
