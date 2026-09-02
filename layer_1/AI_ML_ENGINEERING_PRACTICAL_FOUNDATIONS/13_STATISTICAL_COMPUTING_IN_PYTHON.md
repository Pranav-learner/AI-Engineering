# 🧮 Module 13: Statistical Computing in Python

---

## 1. Computing Descriptive Statistics from Scratch in NumPy

```python
import numpy as np
from scipy import stats

data = np.array([1200, 1450, 1300, 1600, 1250, 85000, 1400])

# Mean & Median
mean = float(np.mean(data))
median = float(np.median(data))

# Variance & Standard Deviation (ALWAYS use ddof=1 for sample statistics)
variance = float(np.var(data, ddof=1))
std_dev = float(np.std(data, ddof=1))

# Covariance & Pearson Correlation Matrix
cov_matrix = np.cov(data, ddof=1)
```

---

## 2. Robust Outlier Detection Engine

```python
def robust_outlier_filter(data: np.ndarray):
    # Median Absolute Deviation
    median = np.median(data)
    mad = np.median(np.abs(data - median))
    if mad == 0: return np.zeros(len(data), dtype=bool)
    mod_z = 0.6745 * np.abs(data - median) / mad
    return mod_z > 3.5

print("Outlier Indices via MAD:", np.where(robust_outlier_filter(data))[0])  # [5] (85k shock!)
```

---

## 3. Student-t Confidence Intervals & Hypothesis Testing

```python
def student_t_confidence_interval(samples: np.ndarray, confidence: float = 0.95):
    n = len(samples)
    mean = np.mean(samples)
    sem = stats.sem(samples)  # s / sqrt(n)
    t_crit = stats.t.ppf((1 + confidence) / 2.0, df=n - 1)
    margin = t_crit * sem
    return {"mean": mean, "ci_lower": mean - margin, "ci_upper": mean + margin}
```
