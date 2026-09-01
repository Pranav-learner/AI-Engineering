import torch 
import torch.nn as nn

class TransformerBlock(nn.Module):

    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()

        self.attention = nn.MultiheadAttention(
            embed_dim = d_model,
            num_heads = n_heads,
            batch_first=True
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model)
        )

    def forward(self, x, causal_mask=None):

        attention_output, _ = self.attention(
            x,
            x,
            x,
            attn_mask=causal_mask
        )

        x = self.norm1(x + attention_output)

        ffn_output = self.ffn(x)

        x = self.norm2(x + ffn_output)

        return x

'''Input
 ↓
Self-Attention
 ↓
Residual + LayerNorm
 ↓
FFN
 ↓
Residual + LayerNorm
 ↓
Output'''

batch_size = 2
sequence_length = 5
d_model = 64
n_heads = 4
d_ff = 256

x = torch.randn(
    batch_size,
    sequence_length,
    d_model
)

block = TransformerBlock(
    d_model,
    n_heads,
    d_ff
)

output = block(x)

print(output.shape)