# 🛡️ Module 27: Model Regularization & Generalization

---

## 1. Techniques to Prevent Overfitting

```
┌────────────────────────────────────────────────────────┐
│ 1. Dropout (nn.Dropout(p=0.15))                        │
│    Randomly disables neurons during training           │
├────────────────────────────────────────────────────────┤
│ 2. Batch Normalization (nn.BatchNorm1d(64))            │
│    Normalizes layer activations; adds slight noise     │
├────────────────────────────────────────────────────────┤
│ 3. Weight Decay (L2 Penalty)                           │
│    Adds α ||w||² to loss function to keep weights small│
├────────────────────────────────────────────────────────┤
│ 4. Early Stopping                                      │
│    Halts training when validation loss stops improving │
└────────────────────────────────────────────────────────┘
```

---

## 2. Early Stopping Implementation

```python
class EarlyStopping:
    def __init__(self, patience: int = 10, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float("inf")
        self.should_stop = False

    def check(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop
```
