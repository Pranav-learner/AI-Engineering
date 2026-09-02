# 🎮 Module 30: AI/ML Interactive Coding Playground

---

## 1. The 5-Step Playground Loop

Whenever you encounter a new operation in Pandas, NumPy, Scikit-Learn, or PyTorch:

```
┌───────────────────────────────┐
│ 1. PREDICT BEFORE RUNNING     │  ──► Guess what the output shape, type, and values will be.
├───────────────────────────────┤
│ 2. RUN CODE                   │  ──► Execute the operation.
├───────────────────────────────┤
│ 3. OBSERVE OUTPUT             │  ──► Check type(result), result.shape, result.head().
├───────────────────────────────┤
│ 4. EXPLAIN THE MECHANICS      │  ──► Why did it behave this way?
├───────────────────────────────┤
│ 5. EXPERIMENT / BREAK IT      │  ──► Change one parameter (e.g. axis=0 vs axis=1) and re-run.
└───────────────────────────────┘
```

---

## 2. Interactive Hands-on Exercises

### Exercise 1: Shape Detective
```python
# Predict before running:
a = np.array([1, 2, 3])
b = a.reshape(1, -1)
c = a.reshape(-1, 1)

print("a.shape:", a.shape)  # Prediction: (3,)
print("b.shape:", b.shape)  # Prediction: (1, 3)
print("c.shape:", c.shape)  # Prediction: (3, 1)
```

### Exercise 2: GroupBy Transform vs. Agg Detective
```python
df = pd.DataFrame({"group": ["A", "A", "B"], "val": [10, 20, 30]})
print("Agg Output Shape      :", df.groupby("group")["val"].agg("mean").shape)        # (2,)
print("Transform Output Shape:", df.groupby("group")["val"].transform("mean").shape)  # (3,)
```
