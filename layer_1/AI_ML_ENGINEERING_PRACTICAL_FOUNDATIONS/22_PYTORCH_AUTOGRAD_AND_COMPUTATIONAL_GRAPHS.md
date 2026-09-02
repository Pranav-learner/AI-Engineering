# 🔄 Module 22: PyTorch Autograd & Computational Graphs

---

## 1. CONCEPT: Dynamic Computational Graphs

When `requires_grad=True`, PyTorch records every mathematical operation in a directed acyclic graph (DAG):

```
x = 2.0 (requires_grad=True)
  │
  ▼
y = x² + 3x + 5   ──► Forward Pass
  │
  ▼
dy/dx = 2x + 3 = 2(2) + 3 = 7.0 ──► Computed via y.backward()
```

```python
import torch

x = torch.tensor(2.0, requires_grad=True)
y = (x ** 2) + (3 * x) + 5

# Backward Pass (computes gradient dy/dx)
y.backward()

print("y value       :", y.item())      # 15.0
print("Gradient dy/dx:", x.grad.item())  # 7.0
```

---

## 2. The Golden Requirement: `optimizer.zero_grad()`

In PyTorch, **gradients accumulate by default** (`param.grad += new_grad`). If you forget to reset gradients at the start of each training batch, the model will add gradients from the current batch to the previous batch, leading to exploding updates and training failure!

```python
optimizer.zero_grad()  # ⚠️ ALWAYS call this before loss.backward()!
loss.backward()
optimizer.step()
```
