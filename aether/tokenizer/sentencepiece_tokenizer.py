import re
import unicodedata
from aether.tokenizer.base_tokenizer import BaseTokenizer

class SentencePieceTokenizer(BaseTokenizer):
    def __init__(self, vocab_size=8000, model_type="unigram", use_gpu=False):
        super().__init__(vocab_size)
        self.model_type = model_type
        self.special_space = "▁"
        self.use_gpu = use_gpu

    def normalize(self, text):
        text = unicodedata.normalize("NFKC", text)
        text = text.lower()
        text = re.sub(r"\s+", self.special_space, text)
        return text

    def train(self, data):
        normalized = [self.normalize(t) for t in data]
        if self.model_type == "unigram":
            from aether.tokenizer.unigram_tokenizer import UnigramTokenizer
            self.subtokenizer = UnigramTokenizer(vocab_size=self.vocab_size, use_gpu=self.use_gpu)
        elif self.model_type == "bpe":
            from aether.tokenizer.bpe_tokenizer import BPETokenizer
            self.subtokenizer = BPETokenizer(vocab_size=self.vocab_size, use_gpu=self.use_gpu)
        else:
            raise ValueError("Unsupported model_type: choose 'unigram' or 'bpe'")

        self.subtokenizer.train(normalized)
        self.token2id = self.subtokenizer.token2id
        self.id2token = self.subtokenizer.id2token

    def encode(self, text):
        normalized = self.normalize(text)
        return self.subtokenizer.encode(normalized)

    def decode(self, ids):
        text = self.subtokenizer.decode(ids)
        return text.replace(self.special_space, " ")