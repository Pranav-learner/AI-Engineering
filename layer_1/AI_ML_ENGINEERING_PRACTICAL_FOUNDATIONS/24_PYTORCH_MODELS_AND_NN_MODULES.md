# 🧠 Module 24: PyTorch Models & `nn.Module` Architecture

---

## 1. CONCEPT: The Anatomy of `nn.Module`

Every neural network in PyTorch inherits from `nn.Module` and implements:
1. **`__init__()`**: Instantiates layers, activation functions, and weights.
2. **`forward(x)`**: Defines the dataflow from input tensor to output prediction.

```python
import torch
import torch.nn as nn

class FinancialRiskMLP(nn.Module):
    def __init__(self, in_features: int, num_classes: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(p=0.15),

            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(p=0.10),

            nn.Linear(32, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
```

---

## 2. The Critical Modes: `model.train()` vs. `model.eval()`

```
┌────────────────────────────────────────────────────────┐
│ model.train() ──► BatchNorm computes batch statistics; │
│                   Dropout randomly zeros neurons       │
├────────────────────────────────────────────────────────┤
│ model.eval()  ──► BatchNorm freezes running stats;     │
│                   Dropout is DISABLED (deterministic)  │
└────────────────────────────────────────────────────────┘
```
