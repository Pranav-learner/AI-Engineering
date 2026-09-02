# 🎯 Module 06: Pandas Indexing, Selection & Filtering

---

## 1. CONCEPT: `.loc[]` vs. `.iloc[]`

```
┌────────────────────────────────────────────────────────┐
│ .loc[]  ──► Label-Based Selection (Names & Row Labels) │
├────────────────────────────────────────────────────────┤
│ .iloc[] ──► Integer-Position Selection (0, 1, 2, ...)  │
└────────────────────────────────────────────────────────┘
```

```python
import pandas as pd

df = pd.DataFrame({
    "income": [90000, 65000, 180000],
    "expense": [25000, 48000, 160000]
}, index=["user_a", "user_b", "user_c"])
```

### 1.1 `.loc[]` Examples (Labels)
```python
# Select Row by Label
print(df.loc["user_a"])             # Series of user_a values

# Select Specific Cell (Row Label, Column Label)
print(df.loc["user_a", "income"])   # 90000

# Select All Rows for specific column list
print(df.loc[:, ["income", "expense"]])
```

### 1.2 `.iloc[]` Examples (Integer Positions)
```python
# Select First Row (Position 0)
print(df.iloc[0])                   # user_a row

# Select First Row, First Column
print(df.iloc[0, 0])                # 90000

# Select First 2 Rows, All Columns
print(df.iloc[:2, :])
```

---

## 2. Boolean Filtering (Data Subsetting)

```python
df = pd.DataFrame({
    "user": ["U1", "U2", "U3", "U4"],
    "income": [90000, 65000, 180000, 35000],
    "expense": [25000, 48000, 160000, 38000],
    "category": ["Saver", "Balanced", "Spender", "Vulnerable"]
})

# 1. Single Condition
high_income_df = df[df["income"] >= 90000]

# 2. Compound AND Condition (&)
vulnerable_df = df[(df["income"] < 50000) & (df["expense"] > 35000)]

# 3. Compound OR Condition (|)
target_cohort = df[(df["category"] == "Saver") | (df["category"] == "Vulnerable")]

# 4. Membership Filtering (.isin())
selected_users = df[df["user"].isin(["U1", "U3"])]

# 5. Fast Query Syntax (.query())
fast_subset = df.query("income > 60000 and expense < 50000")
```

---

## 3. The `SettingWithCopyWarning` Trap & Safe Assignment

### The Problem:
```python
# Chained indexing triggers SettingWithCopyWarning:
subset = df[df["income"] > 50000]
subset["income"] = 100000  # ⚠️ Modifying a view vs a copy is undefined!
```

### The Production Fix: `.loc[]` or `.copy()`
```python
# Option A: Explicit Copy
subset = df[df["income"] > 50000].copy()
subset["income"] = 100000  # Safe!

# Option B: Direct .loc[] assignment on main DataFrame
df.loc[df["income"] > 50000, "is_qualified"] = True  # Safe!
```
