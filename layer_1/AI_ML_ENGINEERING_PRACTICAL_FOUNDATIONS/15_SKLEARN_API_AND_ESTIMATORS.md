# 🏛️ Module 15: Scikit-Learn API & Estimator Architecture

---

## 1. The 4 Fundamental Estimator Methods

```
┌────────────────────────────────────────────────────────┐
│ 1. fit(X, y)         ──► Learns state & parameters     │
│                          from training data            │
├────────────────────────────────────────────────────────┤
│ 2. predict(X)        ──► Predicts target values on     │
│                          unseen samples                │
├────────────────────────────────────────────────────────┤
│ 3. predict_proba(X)  ──► Predicts class probabilities  │
│                          (for classifiers)             │
├────────────────────────────────────────────────────────┤
│ 4. fit_transform(X)  ──► Fits transformer and returns  │
│                          transformed data in 1 call    │
└────────────────────────────────────────────────────────┘
```

---

## 2. The Critical Distinction: `fit()` vs. `transform()` vs. `fit_transform()`

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

# 1. On TRAINING Data: fit_transform()
# Calculates mu_train and sigma_train AND scales X_train:
X_train_scaled = scaler.fit_transform(X_train)

# 2. On TEST Data: ONLY transform()
# Scales X_test using the LEARNED mu_train and sigma_train:
X_test_scaled = scaler.transform(X_test)  # ⚠️ NEVER call fit_transform on X_test!
```

---

## 3. Writing Custom Estimators

```python
from sklearn.base import BaseEstimator, RegressorMixin

class CustomMeanRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, fallback_val: float = 0.0):
        self.fallback_val = fallback_val

    def fit(self, X, y):
        self.learned_mean_ = float(np.mean(y))  # Store learned parameters with trailing underscore
        return self  # Always return self

    def predict(self, X):
        return np.full(len(X), self.learned_mean_)
```
