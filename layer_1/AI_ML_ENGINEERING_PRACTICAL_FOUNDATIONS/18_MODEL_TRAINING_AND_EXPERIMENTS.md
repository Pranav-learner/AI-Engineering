# 🧪 Module 18: Model Training, Experiments & Reproducibility

---

## 1. The Definitions: Epoch, Batch, and Iteration

```
Dataset: 1,000 samples | Batch Size: 50
- 1 Batch     = 50 samples processed at once.
- 1 Iteration = 1 step of gradient descent / weight update (1 batch).
- 1 Epoch     = 1 complete pass through all 1,000 samples (20 iterations = 1 epoch).
```

---

## 2. Seed Reproducibility Contract

Always fix random seeds globally at the top of your project:

```python
import os
import random
import numpy as np
import torch

def seed_everything(seed: int = 42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

seed_everything(42)
```
