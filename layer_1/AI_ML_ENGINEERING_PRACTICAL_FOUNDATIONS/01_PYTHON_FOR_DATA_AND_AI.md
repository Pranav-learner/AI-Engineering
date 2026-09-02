# 🐍 Module 01: Python for Data & AI Engineering

---

## 1. CONCEPT: Python's Object & Memory Model for AI

In Python, **variables do not hold values; variables are pointers/references to objects in memory**.

```
CODE:                           MEMORY HEAP:
a = [10, 20, 30] ───────────►   [ 10, 20, 30 ] (List Object @ 0x7ffd)
                                        ▲
b = a            ───────────────────────┘ (b points to the EXACT same list!)
```

### 1.1 Mutability Trap in ML Pipelines
```python
# The Hidden Bug:
train_features = [1200, 450, 800]
augmented_features = train_features  # Shallow reference!

augmented_features.append(99999)  # Mutates the original list too!
print("Train Features:", train_features)  # [1200, 450, 800, 99999] <- Contaminated!

# The Fix: Explicit Copy
augmented_features = train_features.copy()  # or list(train_features)
```

---

## 2. Advanced Slicing & Indexing (`[start:stop:step]`)

```
Index:    0    1    2    3    4    5    6    7    8    9
Array:  [10,  20,  30,  40,  50,  60,  70,  80,  90, 100]
Neg:    -10   -9   -8   -7   -6   -5   -4   -3   -2   -1
```

```python
data = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

print(data[0])      # 10 (first element)
print(data[-1])     # 100 (last element)
print(data[1:5])    # [20, 30, 40, 50] (indices 1 to 4 inclusive)
print(data[:3])     # [10, 20, 30] (first 3 elements)
print(data[-3:])    # [80, 90, 100] (last 3 elements)
print(data[::2])    # [10, 30, 50, 70, 90] (every 2nd element / downsampling)
print(data[::-1])   # [100, 90, 80, ...] (reverse sequence)
```

---

## 3. List Comprehensions & High-Performance Iteration

### 3.1 Comprehensions vs. Traditional Loops
```python
# Traditional Loop (Slow, Verbose)
squared_spends = []
for x in data:
    if x > 50:
        squared_spends.append(x ** 2)

# List Comprehension (Fast, Clean, Idiomatic Python)
squared_spends = [x ** 2 for x in data if x > 50]
```

### 3.2 `zip()` and `enumerate()` for Batch Processing
```python
user_ids = ["USR_01", "USR_02", "USR_03"]
incomes = [90000, 65000, 180000]
expenses = [25000, 48000, 160000]

# enumerate gives (index, value)
for idx, uid in enumerate(user_ids):
    print(f"Index {idx}: User {uid}")

# zip pairs multiple iterables element-by-element
for uid, inc, exp in zip(user_ids, incomes, expenses):
    savings = inc - exp
    print(f"{uid}: Net Savings = ₹{savings:,}")
```

---

## 4. Functions, `*args`, `**kwargs`, and Lambdas

### 4.1 Positional & Keyword Argument Unpacking
```python
def train_model(model_name: str, *features, lr: float = 0.001, **hyperparams):
    """
    *features captures arbitrary positional arguments into a tuple.
    **hyperparams captures arbitrary keyword arguments into a dictionary.
    """
    print(f"Training {model_name} with lr={lr}")
    print(f"Features passed ({len(features)}):", features)
    print("Hyperparameters dictionary:", hyperparams)

train_model(
    "RandomForest",
    "income", "expenses", "runway", "entropy",  # -> *features tuple
    lr=0.01,
    n_estimators=100, max_depth=6, class_weight="balanced"  # -> **hyperparams dict
)
```

### 4.2 Lambda Functions (Anonymous 1-Line Functions)
```python
# Syntax: lambda inputs : output_expression
compute_savings_rate = lambda inc, exp: (inc - exp) / inc if inc > 0 else 0.0

print(compute_savings_rate(90000, 25000))  # 0.7222 (72.2%)
```

---

## 5. Modules, Packages, and Imports (`-m` execution)

```
project_root/
├── .venv/                      # Isolated virtual environment
├── package_a/
│   ├── __init__.py             # Marks folder as Python package
│   ├── config.py               # Shared constants
│   └── data_processor.py       # Functions
└── run.py
```

- **Absolute import**: `from package_a.config import RANDOM_SEED`
- **Relative import**: `from .config import RANDOM_SEED` (inside `package_a`)
- **Execution rule**: Always run from project root using:
  ```bash
  python -m package_a.data_processor
  ```

---

## 6. Interview Question
**Q: What happens under the hood when a default argument is a mutable object (e.g. `def append_metric(val, history=[]):`)?**
- **Answer**: In Python, default parameter expressions are evaluated **once at function definition time**, not every time the function is called. If the default is a mutable list `history=[]`, that exact same list object in memory will be shared across all subsequent calls, causing silent state accumulation across independent runs!
- **Fix**: Always use `def append_metric(val, history=None): if history is None: history = []`.
