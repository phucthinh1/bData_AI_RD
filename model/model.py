import torch
import torch.nn as nn
from .transformer_block import TransformerBlock

class GPTMini(nn.Module):
    def __init__(self, vocab_size=2000, seq_len=256, hidden_size=256, num_layers=6, num_heads=8, ffn_size=1024):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, hidden_size)
        self.pos_emb = nn.Parameter(torch.zeros(1, seq_len, hidden_size))
        self.layers = nn.ModuleList([TransformerBlock(hidden_size, num_heads, ffn_size) for _ in range(num_layers)])
        self.ln_f = nn.LayerNorm(hidden_size)
        self.head = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, x):
        x = self.token_emb(x) + self.pos_emb[:, :x.size(1), :]
        for layer in self.layers:
            x = layer(x)
        x = self.ln_f(x)
        logits = self.head(x)
        return logits
