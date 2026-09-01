import torch
import torch.nn as nn
import torch.optim as optim

vocab_size = 1000
embedding_dim = 128

model = nn.Sequential(
    nn.Embedding(vocab_size, embedding_dim),
    nn.Linear(embedding_dim, vocab_size)
)

optimizer = optim.Adam(
    model.parameters(),
    lr=1e-3
)

loss_fn = nn.CrossEntropyLoss()

# Training 
for step in range(1000):

    x = torch.randint(
        0,
        vocab_size,
        (32, 20)
    )

    y = torch.randint(
        0,
        vocab_size,
        (32, 20)
    )

    logits = model(x)

    loss = loss_fn(
        logits.reshape(-1, vocab_size),
        y.reshape(-1)
    )

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    if step % 100 == 0:
        print(step, loss.item())


'''The important mechanics are:

Input
 ↓
Forward
 ↓
Loss
 ↓
Backward
 ↓
Optimizer
 ↓
Parameter update'''