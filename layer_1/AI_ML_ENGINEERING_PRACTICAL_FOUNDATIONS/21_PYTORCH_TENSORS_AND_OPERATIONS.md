# 🔥 Module 21: PyTorch Tensors & Hardware Acceleration

---

## 1. CONCEPT: NumPy `ndarray` vs. PyTorch `Tensor`

A PyTorch **Tensor** is a multidimensional array that:
1. Supports **GPU / TPU acceleration** via CUDA.
2. Tracks computation graphs for **automatic differentiation** (`requires_grad=True`).

```python
import numpy as np
import torch

# 1. Conversion NumPy <-> PyTorch
np_arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
tensor = torch.from_numpy(np_arr)       # Zero-copy memory sharing
back_to_np = tensor.numpy()

# 2. Tensor Creation
t_zeros = torch.zeros((3, 4), dtype=torch.float32)
t_rand = torch.randn((32, 18))  # Batch of 32, 18 features

# 3. Hardware Device Placement (CPU vs GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tensor_on_gpu = t_rand.to(device)

print("Tensor Shape :", tensor_on_gpu.shape)
print("Tensor Dtype :", tensor_on_gpu.dtype)
print("Tensor Device:", tensor_on_gpu.device)
```
