'''1. MLP (Multilayer Perceptron)

Intuition:
A stack of fully connected layers where every neuron in one layer connects to all neurons in the next. It applies linear combinations followed by non-linear activations to partition complex multi-dimensional decision spaces.

Real-World Use Cases:

    Tabular data prediction (e.g., credit risk scoring, customer churn prediction).

    Embedding classification heads inside larger multimodal pipelines.'''

import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, in_features=10, hidden_dim=64, out_classes=2):
        super().__init__()
        self.network = nn.Sequential(
            # 1. Project raw features into a higher-dimensional latent space
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),  # Non-linear thresholding
            # 2. Map latent representations to final class logits
            nn.Linear(hidden_dim, out_classes)
        )

    def forward(self, x):
        # Input shape: (batch_size, in_features) -> Output shape: (batch_size, out_classes)
        return self.network(x)

# Example run:
x = torch.randn(32, 10)
model = MLP()
logits = model(x)  # Shape: (32, 2)


'''2. CNN (Convolutional Neural Network)

Intuition:
Uses learnable sliding filter kernels to exploit spatial hierarchy and translation invariance. Early layers extract low-level edges and textures, while deeper layers combine them into complex objects regardless of where they appear in the grid.

Real-World Use Cases:

    Medical imaging diagnosis (e.g., tumor detection in X-rays/MRI scans).

    Autonomous driving perception (e.g., lane and pedestrian detection)'''

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            # Conv2d: (in_channels=3, out_channels=16, kernel_size=3, padding=1)
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            # MaxPool2d downsamples spatial grid by 2x to reduce dimension and add invariance
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.classifier = nn.Linear(16 * 16 * 16, num_classes)

    def forward(self, x):
        # x shape: (batch_size, 3, 32, 32)
        feat = self.features(x)                  # -> (batch_size, 16, 16, 16)
        flat = feat.flatten(start_dim=1)         # Flatten to vector -> (batch_size, 16*16*16)
        return self.classifier(flat)             # -> (batch_size, num_classes)

# Example run:
img = torch.randn(8, 3, 32, 32)
model = SimpleCNN()
out = model(img)  # Shape: (8, 10)

'''3. RNN (Recurrent Neural Network)Intuition:Processes sequential data step-by-step by passing a recurrent hidden state ($h_t$) from one step to the next. It acts as an internal memory of past inputs, though standard RNNs suffer from vanishing gradients across long time horizons.Real-World Use Cases:Short-horizon sensor telemetry monitoring.Basic sequential time-series anomaly detection.'''

class VanillaRNN(nn.Module):
    def __init__(self, input_size=10, hidden_size=32, num_classes=2):
        super().__init__()
        # batch_first=True expects input shape: (batch_size, seq_len, input_size)
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # out: all hidden states (batch_size, seq_len, hidden_size)
        # h_n: final hidden state (1, batch_size, hidden_size)
        out, h_n = self.rnn(x)
        # Take the last time step's hidden state for sequence classification
        last_step_state = out[:, -1, :] 
        return self.fc(last_step_state)

# Example run:
seq = torch.randn(16, 5, 10)  # batch=16, 5 time steps, 10 features each
model = VanillaRNN()
pred = model(seq)  # Shape: (16, 2)

'''4. LSTM (Long Short-Term Memory)

Intuition:
Solves the vanishing gradient problem in RNNs by introducing an explicit cell state (a conveyor belt of long-term memory) regulated by three multiplicative gates: Forget (what to drop), Input (what to add), and Output (what to expose).

Real-World Use Cases:

    Stock market volatility and algorithmic trading forecasting.

    Speech recognition and acoustic signal processing.'''


class SimpleLSTM(nn.Module):
    def __init__(self, input_dim=8, hidden_dim=64, num_layers=1):
        super().__init__()
        # LSTM tracks both hidden state (h) and cell state (c) internally
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # output shape: (batch_size, seq_len, hidden_dim)
        # (hn, cn): final hidden and cell states
        output, (hn, cn) = self.lstm(x)
        # Predict target value from the most recent hidden state
        return self.fc(output[:, -1, :])

# Example run:
time_data = torch.randn(4, 20, 8)  # 20 time steps
model = SimpleLSTM()
y_hat = model(time_data)  # Shape: (4, 1)


'''5. GRU (Gated Recurrent Unit)

Intuition:
A streamlined, computationally lighter variant of LSTM that merges the cell state and hidden state. It uses only two gates—Reset Gate (how to combine new input with past memory) and Update Gate (how much past memory to retain).

Real-World Use Cases:

    Embedded/Edge device NLP processing (lower memory/compute footprint than LSTM).

    User session clickstream modeling for real-time recommendations.'''

class SimpleGRU(nn.Module):
    def __init__(self, input_dim=16, hidden_dim=48):
        super().__init__()
        # GRU combines forget & input gates into a single update gate
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # out: (batch_size, seq_len, hidden_dim)
        # h_n: (1, batch_size, hidden_dim)
        out, h_n = self.gru(x)
        return self.head(h_n.squeeze(0))

# Example run:
user_actions = torch.randn(8, 12, 16)
model = SimpleGRU()
score = model(user_actions)  # Shape: (8, 1)


''' 6. Encoder–Decoder

Intuition:
A general two-stage framework where an Encoder maps a variable-length input sequence into a 
fixed-length continuous context vector, and a Decoder unfolds that context vector into an output sequence of 
a different length or structure.
Real-World Use Cases:Image captioning (CNN Encoder extracts image features $\rightarrow$ RNN Decoder outputs text).
Text summarization and paraphrasing.'''

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.rnn = nn.GRU(input_dim, hidden_dim, batch_first=True)
        
    def forward(self, x):
        # Compress entire source sequence into final summary vector `hidden`
        _, hidden = self.rnn(x) 
        return hidden  # Shape: (1, batch_size, hidden_dim)

class Decoder(nn.Module):
    def __init__(self, output_dim, hidden_dim):
        super().__init__()
        self.rnn = nn.GRU(output_dim, hidden_dim, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, target_step, hidden):
        # target_step shape: (batch_size, 1, output_dim)
        # Unrolls one step conditioned on the passed context hidden state
        out, hidden = self.rnn(target_step, hidden)
        prediction = self.fc_out(out)
        return prediction, hidden


'''7. Seq2Seq (Sequence-to-Sequence)

Intuition:
A concrete end-to-end architecture built on the Encoder–Decoder pattern. It takes an input sequence in one 
domain/order (e.g., English words) and autoregressively generates a target sequence in another (e.g., French 
words) step-by-step.

Real-World Use Cases:
Machine translation (e.g., English $\rightarrow$ Spanish).
Chatbot conversation generation and code generation.'''

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, trg_len, out_dim):
        batch_size = src.size(0)
        # 1. Encode source into context representation
        hidden = self.encoder(src)
        
        # 2. Initialize first token (e.g., start-of-sequence token zeros)
        dec_input = torch.zeros(batch_size, 1, out_dim)
        outputs = []
        
        # 3. Autoregressively generate next token step-by-step
        for _ in range(trg_len):
            pred, hidden = self.decoder(dec_input, hidden)
            outputs.append(pred)
            dec_input = pred  # Feed current prediction as the next step's input
            
        return torch.cat(outputs, dim=1)  # Shape: (batch_size, trg_len, out_dim)

# Example run:
enc = Encoder(input_dim=10, hidden_dim=32)
dec = Decoder(output_dim=10, hidden_dim=32)
seq2seq = Seq2Seq(enc, dec)
src_seq = torch.randn(4, 7, 10)  # Input length 7
generated = seq2seq(src_seq, trg_len=5, out_dim=10)  # Generates output length 5 -> (4, 5, 10)


'''8. Attention (Scaled Dot-Product Self-Attention)

Intuition:
Eliminates sequential recurrence by allowing every token to look directly at every other token simultaneously. 
It computes dynamic alignment weights between Queries ($Q$) and Keys ($K$) to take a weighted blend of Values ($V$), 
giving direct long-range context without bottlenecking information through a fixed-size vector.

Real-World Use Cases:
Large Language Models (GPT-4, Claude, LLaMA) and Transformers.
Vision Transformers (ViT) for zero-shot object classification.
'''


import math

class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k):
        super().__init__()
        self.d_k = d_k

    def forward(self, Q, K, V, mask=None):
        # Q, K, V shapes: (batch_size, seq_len, d_k)
        
        # 1. Compute dot-product similarity matrix between all queries and keys
        # (batch, seq_len, d_k) @ (batch, d_k, seq_len) -> (batch, seq_len, seq_len)
        scores = torch.bmm(Q, K.transpose(1, 2)) / math.sqrt(self.d_k)
        
        # 2. Optionally mask future tokens (causal attention) or padding
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
            
        # 3. Convert scores to probability distribution (Attention Weights)
        attn_weights = torch.softmax(scores, dim=-1)
        
        # 4. Multiply attention weights with Values to extract context
        # (batch, seq_len, seq_len) @ (batch, seq_len, d_k) -> (batch, seq_len, d_k)
        context = torch.bmm(attn_weights, V)
        return context, attn_weights

# Example run:
seq_len, d_k = 6, 64
q = torch.randn(2, seq_len, d_k)
k = torch.randn(2, seq_len, d_k)
v = torch.randn(2, seq_len, d_k)

attn = ScaledDotProductAttention(d_k)
out_context, weights = attn(q, k, v)
# out_context shape: (2, 6, 64), weights shape: (2, 6, 6)