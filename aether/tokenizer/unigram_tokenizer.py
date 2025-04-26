import torch
import random
from collections import defaultdict, Counter
from aether.tokenizer.base_tokenizer import BaseTokenizer
from aether.tokenizer.utils import fast_split

class UnigramTokenizer(BaseTokenizer):
    def __init__(self, vocab_size=10000, dropout=0.0):
        super().__init__(vocab_size)
        self.dropout = dropout
        self.tokens = set()

    def train(self, data):
        corpus = Counter(fast_split(data))
        for word in corpus:
            for i in range(len(word)):
                for j in range(i + 1, min(len(word), i + 10) + 1):
                    self.tokens.add(word[i:j])

        token_probs = {token: 1.0 for token in self.tokens}
        
        for _ in range(10):
            scores = defaultdict(float)
            for word, freq in corpus.items():
                word_pieces = self._encode_word(word, token_probs)
                for piece in word_pieces:
                    scores[piece] += freq
            token_probs = dict(sorted(scores.items(), key=lambda x: -x[1])[:self.vocab_size])

        self.token2id = {tok: idx for idx, tok in enumerate(token_probs)}
        self.id2token = {idx: tok for tok, idx in self.token2id.items()}

    def _encode_word(self, word, token_probs):
        i = 0
        pieces = []
        while i < len(word):
            matched = None
            for j in range(min(10, len(word) - i), 0, -1):
                sub = word[i:i+j]
                if sub in token_probs and (random.random() > self.dropout):
                    matched = sub
                    break
            if matched:
                pieces.append(matched)
                i += len(matched)
            else:
                pieces.append(word[i])
                i += 1
        return pieces

    def encode(self, text):
        tokens = self._encode_word(text, self.token2id)
        return [self.token2id.get(t, 0) for t in tokens]

    def decode(self, ids):
        return ''.join([self.id2token[i] for i in ids])