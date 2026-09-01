import torch
import torch.nn as nn


class TinyGPT(nn.Module):

    def __init__(
        self,
        vocab_size,
        d_model=128,
        n_heads=4,
        n_layers=4
    ):
        super().__init__()

        self.token_embedding = nn.Embedding(
            vocab_size,
            d_model
        )

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            layer,
            num_layers=n_layers
        )

        self.lm_head = nn.Linear(
            d_model,
            vocab_size
        )

    def forward(self, tokens):

        x = self.token_embedding(tokens)

        x = self.transformer(x)

        logits = self.lm_head(x)

        return logits