import torch
from torch.utils.data import Dataset

class TextDataset(Dataset):
    def __init__(self, file_path, tokenizer, seq_len=16):
        self.seq_len = seq_len
        with open(file_path, 'r', encoding='utf-8') as f:
            self.data = f.read().split('\n')
        self.tokenizer = tokenizer
        self.tokens = []
        for line in self.data:
            self.tokens.extend(self.tokenizer.encode(line))
    
    def __len__(self):
        return max(1, len(self.tokens) // self.seq_len)
    
    def __getitem__(self, idx):
        start = idx * self.seq_len
        end = start + self.seq_len

        x = self.tokens[start:end]
        y = self.tokens[start+1:end+1]

        # cắt hoặc padding để đảm bảo bằng seq_len
        if len(x) < self.seq_len:
            x += [0] * (self.seq_len - len(x))
        if len(y) < self.seq_len:
            y += [0] * (self.seq_len - len(y))

        return torch.tensor(x), torch.tensor(y)
