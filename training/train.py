import torch
from torch.utils.data import DataLoader
from model.model import GPTMini
from training.dataset import TextDataset
from tokenizer.utils import sp
import torch.nn as nn
import torch.optim as optim
import training.config as cfg
import os



# paths
data_path = os.path.join(os.path.dirname(__file__), "..", "data", "train.txt")
checkpoint_dir = os.path.join(os.path.dirname(__file__), "..", "checkpoints")
checkpoint_dir = os.path.abspath(checkpoint_dir) 
os.makedirs(checkpoint_dir, exist_ok=True)

# Dataset
dataset = TextDataset(data_path, tokenizer=sp, seq_len=cfg.seq_len)
dataloader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)

# Model
model = GPTMini(vocab_size=cfg.vocab_size,
                seq_len=cfg.seq_len,
                hidden_size=cfg.hidden_size,
                num_layers=cfg.num_layers,
                num_heads=cfg.num_heads,
                ffn_size=cfg.ffn_size).to(cfg.device)

optimizer = optim.AdamW(model.parameters(), lr=cfg.lr)
criterion = nn.CrossEntropyLoss()


for epoch in range(cfg.epochs):
    for x, y in dataloader:
        x, y = x.to(cfg.device), y.to(cfg.device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits.view(-1, cfg.vocab_size), y.view(-1))
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1} done, Loss={loss.item()}")
    torch.save(model.state_dict(), os.path.join(checkpoint_dir, f"model_epoch{epoch+1}.pt"))

