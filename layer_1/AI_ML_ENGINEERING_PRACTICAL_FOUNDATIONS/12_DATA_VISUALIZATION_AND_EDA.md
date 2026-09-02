# 📈 Module 12: Exploratory Data Analysis & Visualization

---

## 1. The 5 Questions of Exploratory Data Analysis (EDA)

```
1. Distribution:  Are features Gaussian or heavily skewed? (Histograms, KDE)
2. Outliers:      Are there extreme anomalous values? (Boxplots, IQR)
3. Correlation:   Are features redundant/collinear? (Heatmaps, Scatter)
4. Class Balance: Is the classification target imbalanced? (Bar charts, Value Counts)
5. Drift:         Does feature distribution change over time? (Time series plots)
```

---

## 2. Text-Based Terminal & Programmatic EDA Tools

```python
import numpy as np
import pandas as pd

def automated_eda_summary(df: pd.DataFrame, target_col: str):
    print("=" * 60)
    print(f"📊 AUTOMATED EDA REPORT: {len(df):,} Rows x {len(df.columns)} Columns")
    print("=" * 60)

    # 1. Missing Values
    missing = df.isna().sum()
    print("\n--- Missing Values ---")
    print(missing[missing > 0] if missing.sum() > 0 else "None (100% Complete)")

    # 2. Skewness Detection
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    skewness = df[numeric_cols].skew()
    print("\n--- Highly Skewed Features (|skew| > 1.0) ---")
    print(skewness[abs(skewness) > 1.0])

    # 3. Target Distribution
    print(f"\n--- Target Distribution: '{target_col}' ---")
    print(df[target_col].value_counts(normalize=True) * 100.0)
```
