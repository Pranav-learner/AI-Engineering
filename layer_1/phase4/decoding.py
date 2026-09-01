import torch

logits = torch.tensor([
    4.0,
    2.0,
    1.0,
    0.5
])

temperature = 0.8

scaled_logits = logits / temperature

probs = torch.softmax(
    scaled_logits,
    dim=-1
)

token = torch.multinomial(
    probs,
    num_samples=1
)

print(probs)
print(token)

## top k
def top_k_filter(logits, k):

    values, indices = torch.topk(
        logits,
        k
    )

    filtered = torch.full_like(
        logits,
        float("-inf")
    )

    filtered[indices] = values

    return filtered

filtered_logits = top_k_filter(
    logits,
    k=2
)

probs = torch.softmax(
    filtered_logits,
    dim=-1
)

token = torch.multinomial(
    probs,
    1
)