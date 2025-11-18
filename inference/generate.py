import torch
from model.model import GPTMini
from tokenizer.utils import sp
import training.config as cfg
import os

checkpoint_path = os.path.join(os.path.dirname(__file__), "..", "checkpoints", "model_epoch5.pt")
checkpoint_path = os.path.abspath(checkpoint_path)

model = GPTMini(vocab_size=cfg.vocab_size,
                seq_len=cfg.seq_len,
                hidden_size=cfg.hidden_size,
                num_layers=cfg.num_layers,
                num_heads=cfg.num_heads,
                ffn_size=cfg.ffn_size).to(cfg.device)
model.load_state_dict(torch.load(checkpoint_path, map_location=cfg.device))
model.eval()

def generate(prompt, max_len=50):
    tokens = sp.encode(prompt)
    input_ids = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(cfg.device)
    with torch.no_grad():
        for _ in range(max_len):
            logits = model(input_ids)
            next_token = torch.argmax(logits[:, -1, :], dim=-1)
            input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)
    return sp.decode(input_ids[0].tolist())
