# 🔗 Module 10: Merging, Joining & Reshaping

---

## 1. CONCEPT: Relational Merges (`merge()`, `concat()`)

In production systems, data comes from multiple relational tables (e.g. `users_table`, `transactions_table`, `credit_scores_table`).

```
INNER JOIN ──► Only rows with matching keys in BOTH tables
LEFT JOIN  ──► All rows from Left table + matching Right table data (Fill missing with NaN)
OUTER JOIN ──► All rows from BOTH tables
CONCAT     ──► Stack tables Vertically (axis=0) or Horizontally (axis=1)
```

```python
import pandas as pd

users = pd.DataFrame({
    "user_id": ["U1", "U2", "U3"],
    "profile": ["Saver", "Balanced", "Spender"]
})

transactions = pd.DataFrame({
    "tx_id": ["TX1", "TX2", "TX3", "TX4"],
    "user_id": ["U1", "U1", "U2", "U4"],
    "amount": [1200, 450, 3200, 500]
})

# 1. Left Merge (Keep all transactions)
merged_df = pd.merge(transactions, users, on="user_id", how="left")
print(merged_df)

# 2. Vertical Concatenation (Stacking months)
jan_df = pd.DataFrame({"month": [1], "spend": [50000]})
feb_df = pd.DataFrame({"month": [2], "spend": [52000]})
combined_df = pd.concat([jan_df, feb_df], ignore_index=True)
```

---

## 2. Wide vs. Long Formats (`pivot_table` vs `melt`)

```
WIDE FORMAT (1 row per user, columns = categories):
User   Rent    Groceries   Dining
U1     25000   1200        850

LONG / TIDY FORMAT (1 row per transaction event):
User   Category    Amount
U1     Rent        25000
U1     Groceries   1200
U1     Dining      850
```

```python
# 1. Reshape Long -> Wide using pivot_table()
wide_table = merged_df.pivot_table(
    index="user_id",
    columns="profile",
    values="amount",
    aggfunc="sum",
    fill_value=0.0
)
print("Wide Pivot:\n", wide_table)

# 2. Reshape Wide -> Long using melt()
long_table = wide_table.reset_index().melt(
    id_vars="user_id",
    var_name="profile_name",
    value_name="total_spend"
)
print("Long Melt:\n", long_table)
```
