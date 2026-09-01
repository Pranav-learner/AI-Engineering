# Gradient clipping
import torch 

torch.nn.utils.clip_grad_norm_(
    model.parameters(),
    max_norm=1.0,
    error_if_nonfinite=True,
)

model.zero_grad()
optimizer.zero_grad()

## Stocahstic gradient decnet

def loss(w):
    return (w - 5) ** 2


def gradient(w):
    return 2 * (w - 5)


w = 0.0
learning_rate = 0.1

for step in range(20):
    g = gradient(w)

    w = w - learning_rate * g

    print(
        step,
        "w =", round(w, 4),
        "loss =", round(loss(w), 4)
    )

## Momentum

w = 0.0

learning_rate = 0.1
beta = 0.9

velocity = 0.0

for step in range(20):
    g = gradient(w)

    velocity = beta * velocity + g

    w = w - learning_rate * velocity

    print(
        step,
        "w =", round(w, 4),
        "loss =", round(loss(w), 4)
    )

## 
import torch
import torch.nn as nn

model = nn.Linear(10, 1)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

criterion = nn.MSELoss()

for x, y in dataloader:

    optimizer.zero_grad()

    prediction = model(x)

    loss = criterion(prediction, y)

    loss.backward()

    optimizer.step()


## COMPARE SGD vs Momentum vs RMSprop vs Adam

import torch
import torch.nn as nn
import torch.optim as optim

# Create a simple model
model = nn.Linear(10, 1)

# Define the loss function
criterion = nn.MSELoss()

# Define optimizers
sgd = optim.SGD(model.parameters(), lr=0.01)

momentum = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

rmsprop = optim.RMSprop(model.parameters(), lr=0.001)

adam = optim.Adam(model.parameters(), lr=0.001)

# Create dummy data
X = torch.randn(100, 10)
y = torch.randn(100, 1)

# Train for a few epochs
epochs = 10

for epoch in range(epochs):

    # SGD
    sgd.zero_grad()
    prediction = model(X)
    loss = criterion(prediction, y)
    loss.backward()
    sgd.step()

    # Momentum
    momentum.zero_grad()
    prediction = model(X)
    loss = criterion(prediction, y)
    loss.backward()
    momentum.step()

    # RMSprop
    rmsprop.zero_grad()
    prediction = model(X)
    loss = criterion(prediction, y)
    loss.backward()
    rmsprop.step()

    # Adam
    adam.zero_grad()
    prediction = model(X)
    loss = criterion(prediction, y)
    loss.backward()
    adam.step()

    print(
        epoch,
        "SGD loss =", round(loss.item(), 4),
        "Momentum loss =", round(loss.item(), 4),
        "RMSprop loss =", round(loss.item(), 4),
        "Adam loss =", round(loss.item(), 4)
    )
