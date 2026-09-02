# 🧠 Module 00: How to Think with Data

---

## 1. CONCEPT: The Fundamental Mental Model

In AI and Machine Learning, **data is not just text or numbers in a file**. Data is a mathematical mapping of real-world entities into a structured geometric space.

```
REAL-WORLD ENTITY                        MATHEMATICAL DATASET
┌───────────────────────────┐           ┌─────────────────────────────────────────┐
│ Bank Customer             │           │ Row (Observation / Sample)              │
│ Transactions over time    │ ────────► │ Columns (Features / Variables / Signals)│
│ Will they default next mo?│           │ Target (Label / Objective / y)          │
└───────────────────────────┘           └─────────────────────────────────────────┘
```

---

## 2. Terminology Deconstruction

| Term | What It Means | Real Example |
| :--- | :--- | :--- |
| **Observation / Sample / Record** | A single row representing one independent instance. | One bank customer in March 2024. |
| **Feature / Variable / Predictor** | A column representing a measured characteristic or signal. | `monthly_income`, `savings_rate`, `runway_days`. |
| **Target / Label / Ground Truth ($y$)** | The exact outcome the AI model is trying to predict. | Next month's expense (₹) or Risk Tier (0/1/2). |
| **Feature Matrix ($\mathbf{X}$)** | 2D table containing all input features for $N$ samples. | Shape: $(N, d) = (\text{samples}, \text{features})$. |
| **Target Vector ($\mathbf{y}$)** | 1D vector containing true outcomes for the $N$ samples. | Shape: $(N,)$ or $(N, 1)$. |
| **Dimensions / Rank** | Number of axes required to index an element. | 1D (vector), 2D (matrix), 3D (time-series batch). |
| **Shape** | Tuple representing the size along each dimension. | `(100, 5)` means 100 rows, 5 columns. |

---

## 3. Data Structure Types

```
1. Structured Data (Tabular)     2. Semi-Structured (JSON / Logs)    3. Unstructured (Audio, Vision, Text)
┌───────┬────────┬────────┐      {                                   "The quick brown fox..."
│ User  │ Income │ Spend  │        "user_id": "USR_01",              [Image Pixels: 3 x 224 x 224]
├───────┼────────┼────────┤        "txs": [{"amt": 450}, ...]        [Audio Waveforms: 16kHz]
│ U_01  │ 90,000 │ 25,000 │      }
└───────┴────────┴────────┘
```

---

## 4. The Shape & Dimensionality Mental Model

### 4.1 1D Data: The Vector ($\mathbf{y} \in \mathbb{R}^N$)
A single sequence of values (e.g. target labels, single column, single time series).
```python
import numpy as np

y = np.array([0, 1, 1, 0, 2])
print("Data:", y)
print("Shape:", y.shape)  # (5,) -> 1 axis, length 5
print("Dimension:", y.ndim) # 1
```

### 4.2 2D Data: The Feature Matrix ($\mathbf{X} \in \mathbb{R}^{N \times d}$)
A table of $N$ observations, where each observation is described by $d$ distinct features.
```python
X = np.array([
    [90000.0, 25000.0, 65000.0, 0.72],
    [65000.0, 48000.0, 17000.0, 0.26],
    [180000.0, 160000.0, 20000.0, 0.11],
    [35000.0, 38000.0, -3000.0, -0.08],
    [85000.0, 30000.0, 55000.0, 0.64],
])

print("Feature Matrix X:\n", X)
print("Shape:", X.shape)      # (5, 4) -> 5 samples (rows), 4 features (columns)
print("Rows (N samples):", X.shape[0])
print("Cols (d features):", X.shape[1])
```

### 4.3 3D & 4D Data: Sequential & Computer Vision Tensors
- **Time-Series / NLP Batch**: `(Batch_Size, Time_Steps, Feature_Dim)` e.g. `(32, 12, 18)` $\implies$ 32 users, 12 historical months, 18 financial metrics per month.
- **Computer Vision Batch**: `(Batch_Size, Channels, Height, Width)` e.g. `(32, 3, 224, 224)` $\implies$ 32 images, 3 color channels (RGB), $224 \times 224$ pixels.

---

## 5. The Core Machine Learning Data Contract: $\mathbf{X}$ and $\mathbf{y}$

```
┌───────────────────────────────────────────────────────────┐
│ FEATURE MATRIX: X (Shape: N x d)                          │
│ [monthly_income, monthly_expense, runway_days, cv]        │
├───────────────────────────────────────────────────────────┤
│ Sample 0: [ 90000.0,  25000.0,  180.0,  0.08 ]            │ ──┐
│ Sample 1: [ 65000.0,  48000.0,   45.0,  0.18 ]            │   │
│ Sample 2: [180000.0, 160000.0,   15.0,  0.40 ]            │   │  Model.fit(X, y)
│ Sample 3: [ 35000.0,  38000.0,    6.0,  0.25 ]            │   │  Learns mapping:
│ Sample 4: [ 85000.0,  30000.0,  120.0,  0.10 ]            │   │      X ──► y
└───────────────────────────────────────────────────────────┘   │
                                                                │
┌───────────────────────────────────────────────────────────┐   │
│ TARGET VECTOR: y (Shape: N,)                              │   │
├───────────────────────────────────────────────────────────┤   │
│ Sample 0: 0 (Low Risk)                                    │ ──┘
│ Sample 1: 1 (Medium Risk)                                 │
│ Sample 2: 2 (High Risk)                                   │
│ Sample 3: 2 (High Risk)                                   │
│ Sample 4: 0 (Low Risk)                                    │
└───────────────────────────────────────────────────────────┘
```

---

## 6. Practice & Mastery Exercises

### Level 1 — Basic Intuition
1. Given a dataset of 5,000 housing sales with columns `[square_feet, bedrooms, bathrooms, zip_code, year_built, sale_price]`:
   - What is $N$ (number of samples)?
   - What are the features $\mathbf{X}$ and what is its shape?
   - What is the target $\mathbf{y}$ and what is its shape?

### Level 2 — Shape Diagnostics
Given the following shapes, identify the machine learning problem:
- `X.shape = (1000, 20)`, `y.shape = (1000,)` with continuous float values $\implies$ **Tabular Regression**.
- `X.shape = (5000, 18)`, `y.shape = (5000,)` with integers `{0, 1, 2}` $\implies$ **Multi-Class Classification**.
- `X.shape = (64, 512, 768)` $\implies$ **Transformer / NLP Batch** (64 sentences, 512 tokens, 768 embedding size).

---

## 7. Interview Question
**Q: What is the critical difference between shape `(N,)` and shape `(N, 1)` in Python, and why does passing `(N, 1)` into a Scikit-Learn classifier throw a `DataConversionWarning`?**
- **Answer**: `(N,)` is a rank-1 array (1D vector with no column dimension). `(N, 1)` is a rank-2 matrix with 1 explicit column. Scikit-Learn classifiers expect targets as a 1D contiguous vector of shape `(N,)`. Passing a 2D column matrix `(N, 1)` triggers a warning because the estimator internally has to call `.ravel()` to flatten it before computing loss gradients.
