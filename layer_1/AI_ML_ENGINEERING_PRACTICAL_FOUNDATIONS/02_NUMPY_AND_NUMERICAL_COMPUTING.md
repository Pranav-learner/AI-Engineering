# 🔢 Module 02: NumPy & Numerical Computing Foundations

---

## 1. CONCEPT: The `ndarray` Engine

NumPy's `ndarray` (N-Dimensional Array) is the backbone of all AI computation (PyTorch, TensorFlow, Scikit-Learn, Pandas).

### 1.1 Why NumPy is $50\times - 100\times$ Faster than Python Lists
- **Python List**: Stores an array of pointers to fragmented Python objects scattered across RAM (type overhead, memory indirection, pointer chasing).
- **NumPy Array**: A single, contiguous block of raw C-memory with a fixed data type (`float32`, `float64`, `int32`), executed via vectorized SIMD CPU/GPU hardware instructions.

```
PYTHON LIST:                            NUMPY NDARRAY:
[ ptr0, ptr1, ptr2 ]                    Contiguous Memory Buffer:
   │     │     │                        ┌──────────┬──────────┬──────────┐
   ▼     ▼     ▼                        │ float64  │ float64  │ float64  │
 [1.0] [2.0] [3.0]                      │ (8 bytes)│ (8 bytes)│ (8 bytes)│
 (Fragmented in RAM)                    └──────────┴──────────┴──────────┘
```

---

## 2. Array Creation & Factory Functions

```python
import numpy as np

# 1. From Python Lists
a = np.array([1.0, 2.0, 3.0], dtype=np.float32)

# 2. Zeros & Ones (Allocates memory buffer)
zeros_mat = np.zeros((3, 4), dtype=np.float64)  # 3 rows, 4 columns
ones_mat  = np.ones((5, 2), dtype=np.float32)   # 5 rows, 2 columns

# 3. Step & Continuous Ranges
range_arr = np.arange(0, 10, step=2)            # [0, 2, 4, 6, 8]
linear_sp = np.linspace(0.0, 1.0, num=5)        # [0.0, 0.25, 0.5, 0.75, 1.0]

# 4. Reproducible Random Generation (Modern NumPy 2.x API)
rng = np.random.default_rng(seed=42)
gaussian_weights = rng.normal(loc=0.0, scale=1.0, size=(100, 18))
uniform_samples  = rng.uniform(low=0.0, high=1.0, size=50)
```

---

## 3. Core Array Attributes: The 4 Invariants

For any array `arr`, always inspect:

```python
arr = np.random.randn(32, 18)

print("arr.shape :", arr.shape)  # (32, 18) -> (N samples, d features)
print("arr.ndim  :", arr.ndim)   # 2        -> 2 axes (rank 2)
print("arr.dtype :", arr.dtype)  # float64  -> 8 bytes per number
print("arr.size  :", arr.size)   # 576      -> total elements (32 * 18 = 576)
print("arr.itemsize:", arr.itemsize) # 8 bytes
print("Total RAM :", arr.size * arr.itemsize, "bytes")  # 4,608 bytes
```

---

## 4. Understanding Dimensions & Axes

```
                    axis=1 (Columns / Across horizontal features ──►)
                    Col 0      Col 1      Col 2
         ┌─────────┬──────────┬──────────┬──────────┐
  Row 0  │ arr[0]  │   1.0    │   2.0    │   3.0    │
axis=0   ├─────────┼──────────┼──────────┼──────────┤
(Rows /  │ arr[1]  │   4.0    │   5.0    │   6.0    │
Down     ├─────────┼──────────┼──────────┼──────────┤
Vertical)│ arr[2]  │   7.0    │   8.0    │   9.0    │
  ▼      └─────────┴──────────┴──────────┴──────────┘
```

- **`axis=0`**: Collapse along rows $\implies$ reduces vertical dimension, returning **1 value per column**.
- **`axis=1`**: Collapse along columns $\implies$ reduces horizontal dimension, returning **1 value per row**.

```python
mat = np.array([
    [1.0, 2.0, 3.0],
    [4.0, 5.0, 6.0],
    [7.0, 8.0, 9.0]
])

# Mean across axis=0 (Column-wise means)
print(np.mean(mat, axis=0))  # [4.0, 5.0, 6.0] -> Shape: (3,)

# Mean across axis=1 (Row-wise means)
print(np.mean(mat, axis=1))  # [2.0, 5.0, 8.0] -> Shape: (3,)
```

---

## 5. Memory Layout: C-Contiguous vs. Fortran-Contiguous

- **C-Order (Row-Major)**: Default in NumPy/PyTorch. Elements in the same row are stored adjacent in memory. Fast when iterating along rows (`axis=1`).
- **Fortran-Order (Column-Major)**: Elements in the same column are stored adjacent.

```python
arr = np.zeros((100, 100), order='C')
print(arr.flags['C_CONTIGUOUS'])  # True
```

---

## 6. Common Mistakes & Debugging
- **Mismatch in dtypes during PyTorch conversion**: PyTorch models default to `torch.float32`. NumPy defaults to `float64`. Passing `float64` directly into PyTorch layers throws:
  `RuntimeError: expected scalar type Float but found Double`.
  - **Fix**: Cast NumPy array: `arr = arr.astype(np.float32)`.
