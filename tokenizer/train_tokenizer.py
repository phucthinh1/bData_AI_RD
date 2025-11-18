import sentencepiece as spm
import os

# Đường dẫn tuyệt đối đến train.txt
base_path = os.path.dirname(__file__)       # folder tokenizer/
data_path = os.path.join(base_path, "..", "data", "train.txt")

spm.SentencePieceTrainer.Train(
    input=data_path,
    model_prefix=os.path.join(base_path, "tokenizer"),
    vocab_size=100,
    model_type='bpe'
)
