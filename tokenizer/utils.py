import os
import sentencepiece as spm

# Lấy folder hiện tại của file utils.py
base_path = os.path.dirname(__file__)  

# Nối đường dẫn tuyệt đối đến tokenizer.model
model_path = os.path.join(base_path, "tokenizer.model")

# Load tokenizer
sp = spm.SentencePieceProcessor(model_file=model_path)

def encode(text):
    return sp.encode(text, out_type=int)

def decode(tokens):
    return sp.decode(tokens)
