# ⚙️ Module 08: Data Transformation & Feature Engineering

---

## 1. Vectorized Operations vs. `.apply()` and `.map()`

```
┌────────────────────────────────────────────────────────┐
│ Vectorized (Native C/SIMD)  ──► 100x Faster (Default)  │
├────────────────────────────────────────────────────────┤
│ .map()                      ──► Fast for Dictionaries  │
├────────────────────────────────────────────────────────┤
│ .apply()                    ──► Flexible for Complex   │
│                                 Row-by-Row Logic       │
└────────────────────────────────────────────────────────┘
```

```python
import numpy as np
import pandas as pd

df = pd.DataFrame({
    "income": [90000, 65000, 180000, 35000],
    "expense": [25000, 48000, 160000, 38000],
    "status": ["active", "pending", "active", "closed"]
})

# 1. Vectorized Operation (Fastest)
df["savings_rate"] = (df["income"] - df["expense"]) / df["income"]

# 2. .map() for Dictionary Replacements
status_map = {"active": 1, "pending": 0, "closed": -1}
df["status_code"] = df["status"].map(status_map)

# 3. .apply() with Custom Function
def categorize_solvency(row):
    if row["income"] > 100000 and row["expense"] < 50000:
        return "High Surplus"
    elif row["income"] < row["expense"]:
        return "Deficit"
    return "Balanced"

df["solvency_class"] = df.apply(categorize_solvency, axis=1)
```

---

## 2. Advanced Numerical Transformations

### 2.1 Clamping (`.clip()`) and Conditional Assignment (`np.where()`)
```python
# Clamp outliers between 0% and 100%
df["savings_rate_clipped"] = df["savings_rate"].clip(lower=-1.0, upper=1.0)

# np.where(condition, if_true, if_false)
df["is_deficit"] = np.where(df["expense"] > df["income"], 1, 0)
```

### 2.2 Binning Continuous Variables (`pd.cut` vs `pd.qcut`)
- **`pd.cut`**: Equal-width bins (e.g. 0-50k, 50k-100k, 100k+).
- **`pd.qcut`**: Equal-frequency quantile bins (e.g. Quartiles: 25%, 50%, 75%).

```python
# Equal width binning
df["income_tier"] = pd.cut(df["income"], bins=[0, 50000, 100000, np.inf], labels=["Low", "Mid", "High"])

# Equal frequency quantile binning
df["income_quartile"] = pd.qcut(df["income"], q=4, labels=["Q1", "Q2", "Q3", "Q4"])
```

### 2.3 Log Transformation for Heavy-Tailed Spends
$$\tilde{x} = \ln(x + 1) = \text{np.log1p}(x)$$
Transforms skewed exponential spending into a Gaussian bell curve for Linear & Neural models.

```python
df["log_income"] = np.log1p(df["income"])
```

---

## 3. Categorical Encoding (One-Hot vs. Ordinal)

```python
# One-Hot Encoding (pd.get_dummies or Sklearn OneHotEncoder)
encoded_df = pd.get_dummies(df, columns=["status"], drop_first=True, dtype=int)
```
