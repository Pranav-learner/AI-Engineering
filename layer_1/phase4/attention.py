import torch
import torch.nn.functional as F

# 3 tokens, embedding dimension = 4
X = torch.tensor([
    [1.0, 0.0, 1.0, 0.0],
    [0.0, 1.0, 0.0, 1.0],
    [1.0, 1.0, 0.0, 0.0]
])

# Simplified: use X directly as Q, K, V
Q = X
K = X
V = X

d_k = K.shape[-1]

scores = Q @ K.T

scaled_scores = scores / (d_k ** 0.5)

weights = F.softmax(scaled_scores, dim=-1)

output = weights @ V

print("Scores:")
print(scores)

print("\nAttention weights:")
print(weights)

print("\nOutput:")
print(output)

'''Pipeline:

X
 ↓
Q K V
 ↓
QKᵀ
 ↓
Scaling
 ↓
Softmax
 ↓
Weights
 ↓
Weights × V
 ↓
Output'''

## casual masking

mask = torch.triu(
    torch.ones(X.shape[0], X.shape[0]),
    diagonal=1
).bool()

scaled_scores = scaled_scores.masked_fill(mask, float("-inf"))

weights = F.softmax(scaled_scores, dim=-1)

output = weights @ V

print(weights)