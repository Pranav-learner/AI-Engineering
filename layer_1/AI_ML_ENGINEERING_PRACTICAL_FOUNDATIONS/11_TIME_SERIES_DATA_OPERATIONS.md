# ⏱️ Module 11: Time-Series Data Operations & Temporal Rules

---

## 1. DateTime Ingestion & Period Extraction

```python
import pandas as pd

df = pd.DataFrame({
    "date": ["2024-01-01", "2024-01-05", "2024-01-06", "2024-02-01"],
    "amount": [90000, 25000, 1200, 92000]
})

df["date"] = pd.to_datetime(df["date"])

# Extract temporal components
df["year"] = df["date"].dt.year
df["month"] = df["date"].dt.month
df["day"] = df["date"].dt.day
df["day_of_week"] = df["date"].dt.dayofweek          # 0=Monday, 6=Sunday
df["is_weekend"] = df["date"].dt.dayofweek >= 5       # Saturday/Sunday flag
df["year_month"] = df["date"].dt.to_period("M")      # '2024-01'
```

---

## 2. Shift Operations: Past vs. Future

```
Row  Month     monthly_expense    .shift(1) (Past Lag)    .shift(-1) (Future Target)
0    Jan 24    50,000             NaN (No past)           52,000 (Feb Spend)
1    Feb 24    52,000             50,000 (Jan)            48,000 (Mar Spend)
2    Mar 24    48,000             52,000 (Feb)            NaN (No future yet)
```

```python
# Past Features (Input Memory)
df["lag_1"] = df["amount"].shift(1)

# Future Target (Supervised Learning Label)
df["target_next_period"] = df["amount"].shift(-1)
```

---

## 3. Rolling & Expanding Windows

```python
# Rolling 3-Period Moving Average (Fixed sliding window)
df["rolling_mean_3"] = df["amount"].rolling(window=3).mean()

# Expanding Average (Cumulative history up to current step)
df["expanding_mean"] = df["amount"].expanding().mean()
```

---

## 4. The Golden Rule: Preventing Temporal Lookahead Leakage
When training an ML model to predict month $t+1$:
- Every feature MUST be computed using data up to timestamp $t$.
- Never compute rolling statistics or scalers that include timestamps $\ge t+1$.
