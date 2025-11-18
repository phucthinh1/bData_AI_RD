import torch

seq_len = 256
batch_size = 16
epochs = 5
lr = 3e-4
vocab_size = 2000
hidden_size = 256
num_layers = 6
num_heads = 8
ffn_size = 1024
device = 'cuda' if torch.cuda.is_available() else 'cpu'
