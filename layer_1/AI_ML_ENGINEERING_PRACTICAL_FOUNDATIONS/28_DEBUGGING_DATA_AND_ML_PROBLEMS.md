# 🔍 Module 28: Debugging Data & ML Engineering Problems

---

## 1. The Systematic AI Debugging Flowchart

When something breaks or metrics look suspicious:

```
1. DATA SHAPES & TYPES
   - Does X.shape match expected (N, d)?
   - Is target y a 1D vector (N,)?
   - Are all features numeric float32/float64?

2. MISSING VALUES & INFS
   - Are there NaNs in X? (df.isna().sum())
   - Are there Inf values caused by division by zero? (np.isinf(X).sum())

3. TARGET LEAKAGE AUDIT
   - Is test accuracy > 99% or R2 > 0.99?
   - Check feature correlation with target (|r| > 0.95).

4. LOSS & GRADIENTS
   - Is loss NaN? (Learning rate too high, unnormalized features).
   - Is loss not decreasing? (Learning rate too low, missing non-linearities).
```

---

## 2. The 5-Line Diagnostic Snippet

Add this assertion block before calling `.fit()`:

```python
def assert_ml_data_health(X: np.ndarray, y: np.ndarray):
    assert not np.isnan(X).any(), "🚨 X contains NaN values!"
    assert not np.isinf(X).any(), "🚨 X contains Inf values!"
    assert len(X) == len(y), f"🚨 Sample count mismatch: len(X)={len(X)} != len(y)={len(y)}"
    assert X.ndim == 2, f"🚨 Feature matrix must be 2D, got shape {X.shape}"
    print("✅ ML Data Health Check Passed! Ready for training.")
```
