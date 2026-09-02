# 🔁 Module 25: Canonical PyTorch Training Loops

---

## 1. The 5 Steps of the Gradient Descent Loop

```
Mini-Batch (batch_X, batch_y)
            │
            ▼
Step 1: optimizer.zero_grad()    ──► Clears old gradients from previous batch
            │
            ▼
Step 2: outputs = model(batch_X) ──► Forward Pass (computes predictions)
            │
            ▼
Step 3: loss = criterion(...)    ──► Computes difference between truth & prediction
            │
            ▼
Step 4: loss.backward()          ──► Autograd backpropagation (computes dLoss/dw)
            │
            ▼
Step 5: optimizer.step()         ──► Updates weights: w = w - lr * grad
```

---

## 2. Complete Production Training & Validation Loop

```python
import torch

def train_and_validate(model, train_loader, val_loader, criterion, optimizer, epochs=50):
    for epoch in range(epochs):
        # 1. Training Phase
        model.train()
        train_loss = 0.0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(batch_X)

        train_loss /= len(train_loader.dataset)

        # 2. Validation Phase (Evaluation Mode)
        model.eval()
        val_loss = 0.0
        with torch.no_grad():  # Disables gradient graph to save RAM/VRAM
            for batch_X, batch_y in val_loader:
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item() * len(batch_X)

        val_loss /= len(val_loader.dataset)
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1:03d}/{epochs:03d}] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
```
