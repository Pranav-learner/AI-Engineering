# 💥 Module 29: The "Break It" Experimentation Lab

---

## 1. Adversarial Experimentation Protocol

To truly master Machine Learning, you must deliberately break your systems and observe how they degrade.

### 1.1 Experiment 1: Injecting Noise & Shocks
```python
def inject_spending_shocks(df: pd.DataFrame, shock_rate: float = 0.20, multiplier: float = 5.0):
    corrupted = df.copy()
    mask = np.random.rand(len(corrupted)) < shock_rate
    corrupted.loc[mask, "monthly_expense"] *= multiplier
    return corrupted
```

### 1.2 Experiment 2: Injecting Data Leakage
```python
def inject_leaky_target(df: pd.DataFrame):
    corrupted = df.copy()
    # Leaky feature: future target + 1% noise
    corrupted["leaky_future_proxy"] = corrupted["target_next_month_expense"] * 1.01
    return corrupted
```

### 1.3 Experiment 3: Extreme Learning Rates in PyTorch
- **$lr = 10.0$**: Exploding gradients, loss becomes `NaN` in Epoch 1.
- **$lr = 10^{-7}$**: Flat loss curve, zero learning after 100 epochs.
- **$lr = 0.005$**: Smooth convergence to sweet spot.
