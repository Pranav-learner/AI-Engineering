# 📐 Module 03: Array Indexing, Slicing & Shapes

---

## 1. CONCEPT: The Colon Operator `:`

The colon `:` means: **"Take all elements along this entire axis without reduction."**

```
Syntax: array[axis_0_selection, axis_1_selection, axis_2_selection, ...]
```

```
Matrix X (Shape: 4 rows x 3 columns):
          Col 0   Col 1   Col 2
Row 0   [  10,     20,     30  ]
Row 1   [  40,     50,     60  ]
Row 2   [  70,     80,     90  ]
Row 3   [ 100,    110,    120  ]
```

---

## 2. Row vs. Column Extraction Operations

### 2.1 Extracting a Single Column across ALL Rows (`X[:, 0]`)
```python
import numpy as np

X = np.array([
    [10, 20, 30],
    [40, 50, 60],
    [70, 80, 90],
    [100, 110, 120]
])

first_col = X[:, 0]  # All rows (:), Column index 0
print("X[:, 0]:", first_col)        # [10, 40, 70, 100]
print("Shape:", first_col.shape)    # (4,) -> 1D vector of length 4
```

### 2.2 Extracting a Single Row across ALL Columns (`X[0, :]`)
```python
first_row = X[0, :]  # Row index 0, All columns (:)
print("X[0, :]:", first_row)        # [10, 20, 30]
print("Shape:", first_row.shape)    # (3,) -> 1D vector of length 3
```

### 2.3 Slicing Sub-Matrices
```python
# Rows 1 to 2, Columns 1 to 2
sub_mat = X[1:3, 1:3]
print("Sub-Matrix:\n", sub_mat)
# [[50, 60],
#  [80, 90]]
print("Shape:", sub_mat.shape)      # (2, 2)
```

---

## 3. Boolean Indexing & Masks (Filtering Records)

A **Boolean Mask** is an array of `True`/`False` values evaluated element-wise:

```python
spends = np.array([1200, 450, 3200, 150, 8500, 400])

# 1. Create Boolean Mask (Condition)
is_high_spend = spends > 1000
print("Mask:", is_high_spend)  # [ True False  True False  True False]

# 2. Filter using Mask
filtered_spends = spends[is_high_spend]
print("Filtered:", filtered_spends)  # [1200, 3200, 8500]

# 3. Multiple Compound Conditions (& = AND, | = OR, ~ = NOT)
moderate = spends[(spends >= 400) & (spends <= 3000)]
print("Moderate Spends:", moderate)  # [1200, 450, 400]
```

---

## 4. Shape Transformations: Reshape, Squeeze, Flatten

```
(100,)        ──► 1D Vector (100 elements)
(100, 1)      ──► 2D Matrix (100 rows, 1 column)
(1, 100)      ──► 2D Matrix (1 row, 100 columns)
(32, 18)      ──► 2D Batch (32 samples, 18 features)
(32, 1, 28, 28) ──► 4D Image Batch (32 images, 1 channel, 28x28 pixels)
```

### 4.1 Reshaping (`.reshape()`)
```python
v = np.arange(12)  # [0, 1, 2, ..., 11], Shape: (12,)

# Reshape to (3, 4)
mat_3x4 = v.reshape(3, 4)
print("3x4 Matrix:\n", mat_3x4)
print("Shape:", mat_3x4.shape)  # (3, 4)

# Reshape using -1 (Automatic dimension inference)
mat_6x2 = v.reshape(6, -1)  # Infers columns = 12 / 6 = 2
print("Shape (6, -1):", mat_6x2.shape)  # (6, 2)
```

### 4.2 Adding & Removing Singleton Dimensions (`np.newaxis`, `squeeze`)
```python
# Convert (N,) to (N, 1) for Scikit-Learn / PyTorch
y_1d = np.array([0, 1, 2, 0, 1])  # Shape: (5,)
y_2d = y_1d[:, np.newaxis]        # Shape: (5, 1) or y_1d.reshape(-1, 1)
print("2D Target Shape:", y_2d.shape)

# Squeeze: Removes all dimensions of size 1
y_squeezed = np.squeeze(y_2d)      # Shape: (5,)
print("Squeezed Shape:", y_squeezed.shape)
```

### 4.3 Matrix Transposition (`.T`)
Flips rows and columns: $\mathbf{X} \in \mathbb{R}^{N \times d} \to \mathbf{X}^T \in \mathbb{R}^{d \times N}$.
```python
X = np.random.randn(100, 18)
print("X shape :", X.shape)    # (100, 18)
print("X.T shape:", X.T.shape) # (18, 100)
```

---

## 5. Interview Question
**Q: Why does `X[0]` return shape `(3,)` for a 2D array `(4, 3)`, but `X[0:1]` returns shape `(1, 3)`?**
- **Answer**: 
  - Integer indexing (`X[0]`) **reduces the rank/dimension** by 1 (extracts the row as a 1D vector).
  - Slice indexing (`X[0:1]`) **preserves the rank** of the original array, returning a 2D sub-matrix with 1 row.
