# 📉 Module 26: Loss Functions & Optimizers

---

## 1. Loss Functions: Expected Shapes & Targets

| Loss Function | PyTorch API | Output Shape $\hat{y}$ | Target Shape $y$ | Task |
| :--- | :--- | :--- | :--- | :--- |
| **Mean Squared Error** | `nn.MSELoss()` | `(N, 1)` Float | `(N, 1)` Float | Expense Regression |
| **Cross-Entropy** | `nn.CrossEntropyLoss()` | `(N, C)` Unnormalized Logits | `(N,)` Integer Class Index | Multi-Class Risk Tier |
| **Binary Cross-Entropy** | `nn.BCEWithLogitsLoss()` | `(N, 1)` Raw Logits | `(N, 1)` Float `{0.0, 1.0}` | Binary Default/Fraud |

### 🚨 Common Shape Trap in `CrossEntropyLoss`:
`nn.CrossEntropyLoss()` expects target class indices as a **1D Long Tensor** of shape `(N,)` containing integers `0, 1, ..., C-1`. It internally applies softmax and computes negative log-likelihood. Do NOT pass one-hot encoded targets or apply `nn.Softmax()` before `CrossEntropyLoss`!

---

## 2. Optimizers: SGD vs. Adam vs. AdamW

- **Adam**: Computes adaptive learning rates using first and second moments ($m_t, v_t$).
- **AdamW**: Decouples weight decay from gradient updates (Standard in modern Deep Learning & Transformers!).

```python
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=0.005,
    weight_decay=1e-4  # L2 penalty on weights
)
```
