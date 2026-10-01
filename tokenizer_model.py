from tokenizers import Tokenizer, models, trainers, pre_tokenizers, processors
from datasets import load_dataset

dataset = load_dataset("maywell/korean_textbooks", "tiny-textbooks", split="train")
texts = [item["text"] for item in dataset]

tokenizer = Tokenizer(models.WordPiece(unk_token="[UNK]"))
tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

trainer = trainers.WordPieceTrainer(
    vocab_size=30522, 
    min_frequency=2,
    special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]
)

tokenizer.train_from_iterator(texts, trainer=trainer)
tokenizer.save("korean_textbooks_tokenizer.json")
