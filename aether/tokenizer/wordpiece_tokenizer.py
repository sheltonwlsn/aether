import re
import torch
from collections import Counter
from aether.tokenizer.base_tokenizer import BaseTokenizer
from aether.tokenizer.utils import fast_split

class WordPieceTokenizer(BaseTokenizer):
    def __init__(self, vocab_size=10000):
        super().__init__(vocab_size)
        self.special_tokens = {"[UNK]": 0}

    def train(self, data):
        corpus = Counter(fast_split(data))
        word_freqs = dict(corpus)

        tokens = set()
        for word in word_freqs:
            tokens.update(word)
        
        while len(tokens) < self.vocab_size:
            candidates = Counter()
            for word, freq in word_freqs.items():
                for i in range(len(word)):
                    for j in range(i+1, len(word)+1):
                        piece = word[i:j]
                        candidates[piece] += freq
            best = max(candidates, key=candidates.get, default=None)
            if not best:
                break
            tokens.add(best)

        tokens.update(self.special_tokens.keys())
        self.token2id = {tok: idx for idx, tok in enumerate(sorted(tokens))}
        self.id2token = {idx: tok for tok, idx in self.token2id.items()}

    def encode(self, text):
        i = 0
        tokens = []
        while i < len(text):
            matched = None
            for j in range(len(text), i, -1):
                sub = text[i:j]
                if sub in self.token2id:
                    matched = sub
                    break
            if matched:
                tokens.append(self.token2id[matched])
                i += len(matched)
            else:
                tokens.append(self.token2id.get("[UNK]", 0))
                i += 1
        return tokens

    def decode(self, ids):
        return ''.join([self.id2token[i] for i in ids])