import torch
import torch.nn as nn

class MLPWithDropout(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=256, output_dim=10, p=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=p),  # 20% chance to drop neurons in this layer
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=p),  # 20% chance here as well
            
            nn.Linear(hidden_dim, output_dim)  # Never put dropout on final output logits
        )

    def forward(self, x):
        return self.net(x)

model = MLPWithDropout(p=0.2)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

# ==========================================
# TRAINING PHASE (Dropout is ACTIVE)
# ==========================================
model.train()  # <-- Activates dropout masks and batchnorm training stats

for batch_x, batch_y in train_loader:
    optimizer.zero_grad()
    
    # During this forward pass:
    # 20% of hidden neurons are randomly zeroed out per sample
    outputs = model(batch_x)
    
    loss = criterion(outputs, batch_y)
    loss.backward()
    optimizer.step()

# ==========================================
# VALIDATION / INFERENCE (Dropout is DISABLED)
# ==========================================
model.eval()  # <-- Disables dropout (all neurons are active and unmasked)

with torch.no_grad():  # <-- Disables gradient computation to save memory/speed
    for val_x, val_y in val_loader:
        val_outputs = model(val_x)  # Fully deterministic full-network prediction
        val_loss = criterion(val_outputs, val_y)



## DROPOUT

import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(20, 64),
            nn.ReLU(),
            nn.Dropout(p=0.3),

            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(p=0.3),

            nn.Linear(32, 2)
        )

    def forward(self, x):
        return self.network(x)

## BATCHNORM

class Network(nn.Module):
    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(20, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),

            nn.Linear(32, 2)
        )

    def forward(self, x):
        return self.network(x)

## WEIGHT DECAY

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-3,
    weight_decay=1e-4
)


## EXPERIMENT  EARLY STOPPING

best_val_loss = float("inf")
patience = 3
bad_epochs = 0

for epoch in range(100):

    train(...)

    val_loss = validate(...)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        bad_epochs = 0

        torch.save(
            model.state_dict(),
            "best_model.pt"
        )

    else:
        bad_epochs += 1

    if bad_epochs >= patience:
        print("Early stopping")
        break