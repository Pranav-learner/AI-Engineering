# 🧼 Module 07: Data Cleaning & Missing Data Handling

---

## 1. Detecting Missing Values: `NaN`, `None`, and `pd.NA`

```python
import numpy as np
import pandas as pd

df = pd.DataFrame({
    "income": [90000, np.nan, 180000, 35000],
    "expense": [25000, 48000, np.nan, 38000],
    "city": ["Mumbai", "Delhi", None, "Bangalore"]
})

# Check boolean mask of missing cells
print(df.isna())

# Count missing values per column
print(df.isna().sum())

# Percentage of missing data per column
print(df.isna().mean() * 100.0)
```

---

## 2. Handling Missing Data: Drop vs. Impute

### 2.1 Dropping Rows or Columns (`.dropna()`)
```python
# Drop any row containing at least 1 missing value
clean_rows = df.dropna()

# Drop rows ONLY if a specific column is NaN (e.g. target column)
clean_target = df.dropna(subset=["income"])

# Drop columns if more than 30% of their data is missing
threshold = len(df) * 0.70
clean_cols = df.dropna(axis=1, thresh=threshold)
```

### 2.2 Imputation Strategies (`.fillna()`)
```python
# 1. Constant Imputation
df["city"] = df["city"].fillna("Unknown")

# 2. Statistical Imputation (Mean / Median)
median_income = df["income"].median()
df["income"] = df["income"].fillna(median_income)

# 3. Time-Series Directional Fill
# ffill = forward-fill (carries last known value forward)
# bfill = backward-fill (carries next value backward)
df["expense"] = df["expense"].ffill().bfill()
```

### 🚨 Critical Machine Learning Leakage Rule:
Never calculate `.mean()` or `.median()` on the **entire dataset** before splitting into Train and Test! 
- Always split `train_df` and `test_df` **first**.
- Calculate `train_median = train_df['income'].median()`.
- Impute `test_df['income'].fillna(train_median)`.

---

## 3. Deduplication & Type Casting

```python
# Deduplication
df = df.drop_duplicates(subset=["user_id", "date"], keep="last")

# Type Conversion (.astype, pd.to_numeric, pd.to_datetime)
df["date"] = pd.to_datetime(df["date"])
df["income"] = pd.to_numeric(df["income"], errors="coerce") # Invalid strings become NaN
df["risk_tier"] = df["risk_tier"].astype(np.int32)
```
