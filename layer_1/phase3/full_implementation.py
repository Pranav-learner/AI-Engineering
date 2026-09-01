## tensor
import torch

x = torch.tensor([
    [1.0, 2.0],
    [3.0, 4.0]
])

# dataset
from torch.utils.data import Dataset

class MyDataset(Dataset):

    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.y[index]


# dataloader turns individuals samples into batches

from torch.utils.data import DataLoader

loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True
)

## why batching?

'''Benefits include:

better hardware utilization
more efficient computation
smoother gradient estimates

But larger batch size also means:

more memory
potentially different optimization behavior

So:

Batch size is an engineering tradeoff, not a magic number.'''

## Real training pipeline


import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

X = torch.randn(1000,10)
y = (X[:,0] + X[:, 1] > 0).long()

dataset = TensorDataset(X,y)

loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True
)

## model
class Classifier(nn.Module):

    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(10, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 2)
        )

    def forward(self, x):
        return self.network(x)

# loss function
loss_fn = nn.CrossEntropyLoss()

# optimizer
model = Classifier()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

## training loop
epochs = 20

for epoch in range(epochs):

    model.train()

    total_loss = 0

    for X_batch, y_batch in loader:

        # Forward pass
        logits = model(X_batch)

        # Calculate loss
        loss = loss_fn(logits, y_batch)

        # Clear old gradients
        optimizer.zero_grad()

        # Backpropagation
        loss.backward()

        # Update parameters
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(loader)

    print(
        f"Epoch {epoch + 1}: "
        f"loss={avg_loss:.4f}"
    )


# The Four Most Important Lines
optimizer.zero_grad()

#Clear previous gradients.

#↓

#logits = model(X_batch)

#Forward pass.

#↓

loss = loss_fn(logits, y_batch)

#Measure error.

#↓

loss.backward()

#Calculate gradients.

#↓

optimizer.step()

#Update parameters.

'''Why zero_grad()?

PyTorch accumulates gradients.

Suppose:

iteration 1
gradient = 0.5

iteration 2
gradient = 0.3

Without clearing:

gradient = 0.8

That's generally not what we want.

So:

optimizer.zero_grad()

resets them before calculating the next update.'''

# Training vs Evaluation Mode

#Two important modes:

model.train()

# and

model.eval()

# Training mode is used during training.

# Evaluation mode is used during validation/inference.

'''This matters for layers such as:

Dropout
BatchNorm'''

# Valdation pipeline

from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model.eval()

correct = 0
total = 0

with torch.no_grad():

    for X_batch, y_batch in val_loader:

        logits = model(X_batch)

        predictions = logits.argmax(dim=1)

        correct += (
            predictions == y_batch
        ).sum().item()

        total += y_batch.size(0)

accuracy = correct / total

print("Validation accuracy:", accuracy)

'''Why torch.no_grad()?

During inference we don't need gradients.

Therefore:

with torch.no_grad():

reduces unnecessary computation and memory usage.

This becomes extremely important later for LLM inference.'''

### COMPLETE PIPLINE

'''Raw Data
   ↓
Train / Validation Split
   ↓
Dataset
   ↓
DataLoader
   ↓
Model
   ↓
Forward Pass
   ↓
Loss
   ↓
Backward Pass
   ↓
Optimizer
   ↓
Updated Model
   ↓
Validation
   ↓
Metrics'''