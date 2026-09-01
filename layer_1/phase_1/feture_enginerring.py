import pandas as pd

data = pd.DataFrame({
    "amount": [1000, 1200, 900, 50000, 1100],
    "avg_amount": [1000, 1000, 1000, 1000, 1000],
    "hour": [10, 14, 9, 2, 11],
    "is_weekend": [0, 0, 0, 1, 1],
    "transactions_24h": [2, 3, 1, 15, 2]
})

print(data)

data["amount_ratio"] = (
    data["amount"] /
    data["avg_amount"]
)

data["high_activity"] = (
    data["transactions_24h"] > 10
).astype(int)

data["night_transaction"] = (
    (data["hour"] < 6) |
    (data["hour"] >= 23)
).astype(int)

print(data)

#Experiment

data["fraud"] = [
    0, 0, 0, 1, 0
]

# Now compare a model using only:
#amount
#against one using:
#amount
#amount_ratio
#transactions_24h
#night_transaction
#is_weekend
#The point isn't whether this tiny dataset produces a meaningful real-world model.
#The point is learning:
#Does adding information actually improve generalization?

# Experminent  Log Transformation

import numpy as np

income = np.random.lognormal(
    mean=10,
    sigma=1.5,
    size=10000
)

print("Mean:", income.mean())
print("Median:", np.median(income))

log_income = np.log1p(income)

print("Transformed mean:", log_income.mean())
print("Transformed median:", np.median(log_income))