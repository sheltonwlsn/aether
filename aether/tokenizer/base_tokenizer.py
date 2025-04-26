import torch
import json
from abc import ABC, abstractmethod

class BaseTokenizer(ABC):
    def __init__(self, vocab_size=10000, special_tokens=None):
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens or {
            "[PAD]": 0,
            "[UNK]": 1,
            "[BOS]": 2,
            "[EOS]": 3
        }
        self.token2id = dict(self.special_tokens)
        self.id2token = {v: k for k, v in self.token2id.items()}

    @abstractmethod
    def train(self, data):
        pass

    @abstractmethod
    def encode(self, text, add_special_tokens=True):
        pass

    @abstractmethod
    def decode(self, ids, skip_special_tokens=True):
        pass

    def save(self, path):
        with open(path, "w") as f:
            json.dump({
                "vocab_size": self.vocab_size,
                "token2id": self.token2id
            }, f)

    def load(self, path):
        with open(path, "r") as f:
            obj = json.load(f)
            self.vocab_size = obj["vocab_size"]
            self.token2id = obj["token2id"]
            self.id2token = {i: t for t, i in self.token2id.items()}