# 🎛️ Module 20: Hyperparameters & Cross-Validation

---

## 1. Parameters vs. Hyperparameters

- **Parameters ($w, b$)**: Learned automatically by the algorithm during training via optimization (e.g. gradient descent, OLS).
- **Hyperparameters ($\alpha$, learning rate, tree depth, $n\_estimators$)**: Structural configuration settings set by the engineer **before** training begins.

---

## 2. Cross-Validation: K-Fold vs. TimeSeriesSplit

```
Standard K-Fold (Random Shuffling):
Fold 1: [ Test ] [ Train ] [ Train ] [ Train ]  <-- ⚠️ Contaminates time series!

TimeSeriesSplit (Expanding Temporal Folds):
Fold 1: [ Train (M1-M6)  ] [ Test (M7)  ]
Fold 2: [ Train (M1-M7)  ] [ Test (M8)  ]
Fold 3: [ Train (M1-M8)  ] [ Test (M9)  ]  <-- ✅ 100% Leak-free!
```

```python
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.ensemble import RandomForestRegressor

tscv = TimeSeriesSplit(n_splits=5)
param_grid = {
    "n_estimators": [50, 100],
    "max_depth": [4, 6, 8]
}

grid_search = GridSearchCV(
    estimator=RandomForestRegressor(random_state=42),
    param_grid=param_grid,
    cv=tscv,
    scoring="neg_mean_absolute_error"
)

grid_search.fit(X_train, y_train)
print("Best Hyperparameters:", grid_search.best_params_)
```
