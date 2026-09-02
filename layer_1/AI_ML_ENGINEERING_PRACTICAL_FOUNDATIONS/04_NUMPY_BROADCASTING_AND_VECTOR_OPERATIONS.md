# ⚡ Module 04: NumPy Broadcasting & Vector Operations

---

## 1. CONCEPT: Element-wise vs. Matrix Multiplication

In Python and PyTorch:
- **`*`**: Element-wise multiplication (Hadamard product: $C_{ij} = A_{ij} \cdot B_{ij}$). Shapes must match or broadcast.
- **`@` (or `np.dot` / `np.matmul`)**: True algebraic matrix multiplication ($\mathbf{C} = \mathbf{A}\mathbf{B}$, where inner dimensions must match: $(N \times d) @ (d \times k) \to (N \times k)$).

```python
import numpy as np

A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

# 1. Element-wise (*)
elem_prod = A * B
# [[ 1*5,  2*6 ],  = [[ 5, 12 ],
#  [ 3*7,  4*8 ]]     [ 21, 32 ]]

# 2. Matrix Multiplication (@)
mat_prod = A @ B
# [[ 1*5 + 2*7,  1*6 + 2*8 ],  = [[ 19, 22 ],
#  [ 3*5 + 4*7,  3*6 + 4*8 ]]     [ 43, 50 ]]
```

---

## 2. The 3 Golden Rules of NumPy & PyTorch Broadcasting

Broadcasting allows arithmetic operations on arrays of **different shapes** without making wasteful copies in RAM.

### 2.1 The Rules
1. Compare dimensions from **right to left** (trailing dimensions first).
2. Two dimensions are compatible if:
   - They are **equal**, OR
   - One of them is **1**.
3. If one array has fewer dimensions, prepend `1`s to the left until ranks match.

---

### 2.2 Visual Demonstration: Matrix `(3, 4)` + Vector `(4,)`

```
Array A:  Shape (3, 4)  ──►  [ [ 1,  2,  3,  4 ],
                               [ 5,  6,  7,  8 ],
                               [ 9, 10, 11, 12 ] ]
                                       +
Array B:  Shape (4,)    ──►  [   10, 20, 30, 40   ]

Step 1: Pad B to left: Shape (1, 4)
Step 2: Stretch B vertically along axis 0 to match (3, 4):
                             [ [ 10, 20, 30, 40 ],
                               [ 10, 20, 30, 40 ],
                               [ 10, 20, 30, 40 ] ]

Result:   Shape (3, 4)  ──►  [ [ 11, 22, 33, 44 ],
                               [ 15, 26, 37, 48 ],
                               [ 19, 30, 41, 52 ] ]
```

```python
A = np.arange(1, 13).reshape(3, 4)
B = np.array([10, 20, 30, 40])  # Shape: (4,)

C = A + B  # Broadcasts automatically!
print("Result Shape:", C.shape)  # (3, 4)
```

---

### 2.3 Real ML Use Case: Feature Normalization (StandardScaler Math)
$$z = \frac{\mathbf{X} - \boldsymbol{\mu}}{\boldsymbol{\sigma}}$$
- $\mathbf{X} \in \mathbb{R}^{N \times d}$ (e.g. $1000 \times 18$)
- $\boldsymbol{\mu} \in \mathbb{R}^{18}$ (Mean of each feature)
- $\boldsymbol{\sigma} \in \mathbb{R}^{18}$ (Std dev of each feature)

```python
X = np.random.randn(1000, 18) * 50 + 1000  # Raw features

# Compute mean and std along axis=0 (across rows)
mu = np.mean(X, axis=0)   # Shape: (18,)
sigma = np.std(X, axis=0) # Shape: (18,)

# Broadcasting: (1000, 18) - (18,) -> (1000, 18)
X_scaled = (X - mu) / (sigma + 1e-8)
print("Normalized Shape:", X_scaled.shape)  # (1000, 18)
print("New Mean:", np.mean(X_scaled, axis=0)[:3]) # [0., 0., 0.]
```

---

## 3. Reductions: `sum`, `mean`, `min`, `max` across `axis`

```python
X = np.array([
    [10, 20],
    [30, 40],
    [50, 60]
])

# axis=0 (Col-wise reduction): collapses rows
print("Col Sums (axis=0):", np.sum(X, axis=0))  # [90, 120] -> Shape: (2,)

# axis=1 (Row-wise reduction): collapses columns
print("Row Sums (axis=1):", np.sum(X, axis=1))  # [30, 70, 110] -> Shape: (3,)

# keepdims=True (Preserves rank-2 matrix shape for safe broadcasting)
row_sums_2d = np.sum(X, axis=1, keepdims=True)
print("keepdims=True Shape:", row_sums_2d.shape) # (3, 1)
```
