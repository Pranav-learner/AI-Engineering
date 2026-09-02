# 📦 Module 23: PyTorch Datasets & DataLoaders

---

## 1. CONCEPT: Batching & Memory Pipelines

```
Raw Pandas / NumPy Table
         │
         ▼
PyTorch Dataset (Implements __len__ and __getitem__)
         │
         ▼
PyTorch DataLoader (Handles batch_size, shuffling, multi-process workers)
         │
         ▼
Mini-Batch (e.g. 32 samples) fed to GPU
```

---

## 2. Standard `Dataset` and `DataLoader` Code

```python
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class TabularFinanceDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]

# Instantiate Loader
dataset = TabularFinanceDataset(X_train_scaled, y_train)
train_loader = DataLoader(dataset, batch_size=16, shuffle=True)

for batch_idx, (batch_X, batch_y) in enumerate(train_loader):
    print(f"Batch {batch_idx}: X shape = {batch_X.shape}, y shape = {batch_y.shape}")
    break
```
