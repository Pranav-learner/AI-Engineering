# 🐼 Module 05: Pandas DataFrames Foundations

---

## 1. CONCEPT: Series vs. DataFrame

Pandas is the tabular manipulation library built on top of NumPy arrays.
- **`Series`**: A 1D labelled array (a single column with an index).
- **`DataFrame`**: A 2D labelled tabular structure (a collection of Series sharing a common index).

```
INDEX           SERIES: "income"                   DATAFRAME: [income, expense, runway]
┌───────┐      ┌────────────────┐         ┌───────┬────────┬─────────┬────────┐
│   0   │ ──►  │    90,000.0    │         │ Index │ Income │ Expense │ Runway │
├───────┤      ├────────────────┤         ├───────┼────────┼─────────┼────────┤
│   1   │ ──►  │    65,000.0    │         │   0   │ 90,000 │ 25,000  │ 180.0  │
├───────┤      ├────────────────┤         │   1   │ 65,000 │ 48,000  │  45.0  │
│   2   │ ──►  │   180,000.0    │         │   2   │180,000 │160,000  │  15.0  │
└───────┘      └────────────────┘         └───────┴────────┴─────────┴────────┘
```

---

## 2. The Crucial Distinction: `df["col"]` vs. `df[["col"]]`

This is the **#1 source of bugs** when passing data into Scikit-Learn or PyTorch!

```python
import pandas as pd

df = pd.DataFrame({
    "user_id": ["U1", "U2", "U3"],
    "income": [90000, 65000, 180000],
    "expense": [25000, 48000, 160000]
})

# 1. Single Bracket: df["income"]
s = df["income"]
print("Type :", type(s))       # <class 'pandas.core.series.Series'>
print("Shape:", s.shape)       # (3,) -> 1D Vector

# 2. Double Bracket: df[["income"]]
sub_df = df[["income"]]
print("Type :", type(sub_df))  # <class 'pandas.core.frame.DataFrame'>
print("Shape:", sub_df.shape)  # (3, 1) -> 2D Feature Matrix!
```

### 🚨 Machine Learning Implication:
- **`model.fit(X, y)`**:
  - `X` expects a **2D Matrix / DataFrame** $\implies$ Pass `df[["income", "expense"]]` or `df[FEATURE_COLUMNS]`.
  - `y` expects a **1D Vector / Series** $\implies$ Pass `df["target_risk_tier"]`.

---

## 3. Essential Inspection Methods

Always run these 5 commands on any new dataset:

```python
print(df.shape)      # (Rows, Columns)
print(df.columns)    # Index list of column headers
print(df.dtypes)     # Data type of each column (int64, float64, object, datetime64)
df.info()            # Summary of memory usage, non-null counts, and dtypes
df.describe()        # 8-point statistical summary (count, mean, std, min, 25%, 50%, 75%, max)
```

---

## 4. Column Creation & Vectorized Assignment

```python
# Create new columns using vectorized arithmetic (No loops!)
df["net_savings"] = df["income"] - df["expense"]
df["savings_rate"] = df["net_savings"] / df["income"]
df["is_high_earner"] = df["income"] >= 100000  # Boolean Series

print(df)
```
