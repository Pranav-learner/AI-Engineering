# 👥 Module 09: GroupBy, Aggregations & Group Features

---

## 1. CONCEPT: The Split-Apply-Combine Pattern

```
Raw Data Table
      │
      ▼
┌─────────────┐
│    SPLIT    │  ──► Divides table into independent groups based on keys (e.g. user_id)
└──────┬──────┘
       │
       ▼
┌─────────────┐
│    APPLY    │  ──► Computes a function (sum, mean, std) on each individual group
└──────┬──────┘
       │
       ▼
┌─────────────┐
│   COMBINE   │  ──► Assembles group results back into a clean Series or DataFrame
└─────────────┘
```

---

## 2. Multi-Metric Aggregations (`.agg()`)

```python
import pandas as pd

df = pd.DataFrame({
    "user_id": ["U1", "U1", "U1", "U2", "U2"],
    "category": ["Rent", "Groceries", "Dining", "Rent", "Groceries"],
    "amount": [25000, 1200, 850, 18000, 950]
})

# 1. Single Column Aggregations
user_stats = df.groupby("user_id")["amount"].agg(["count", "mean", "std", "max", "sum"])
print(user_stats)

# 2. Named Multi-Column Aggregations (Production Syntax)
user_summary = df.groupby("user_id").agg(
    total_spend=("amount", "sum"),
    avg_spend=("amount", "mean"),
    max_single_spend=("amount", "max"),
    unique_categories=("category", "nunique")
).reset_index()

print(user_summary)
```

---

## 3. The Critical Distinction: `.agg()` vs. `.transform()`

```
┌────────────────────────────────────────────────────────┐
│ .agg()       ──► REDUCES rows (1 output row per group) │
├────────────────────────────────────────────────────────┤
│ .transform() ──► PRESERVES original row count          │
│                  (Broadcasts group result to every row)│
└────────────────────────────────────────────────────────┘
```

```python
# 1. Using .agg() -> Output Shape: (2, 1) -> 2 unique users
agg_result = df.groupby("user_id")["amount"].agg("mean")
print("Agg Result:\n", agg_result)

# 2. Using .transform() -> Output Shape: (5,) -> Matches original 5 rows!
df["user_mean_spend"] = df.groupby("user_id")["amount"].transform("mean")
print("With Transform:\n", df)

# Now we can compute deviation from personal average in 1 line:
df["spend_deviation"] = df["amount"] - df["user_mean_spend"]
```
