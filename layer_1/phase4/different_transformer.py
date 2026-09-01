## ENCODER

import torch
import torch.nn as nn

encoder_layer = nn.TransformerEncoderLayer(
    d_model=128,
    nhead=8,
    batch_first=True
)

encoder = nn.TransformerEncoder(
    encoder_layer,
    num_layers=4
)

## DECODER

decoder_layer = nn.TransformerDecoderLayer(
    d_model=128,
    nhead=8,
    batch_first=True
)

decoder = nn.TransformerDecoder(
    decoder_layer,
    num_layers=4
)