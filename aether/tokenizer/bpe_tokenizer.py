import re
from collections import defaultdict, Counter
import torch
from aether.tokenizer.base_tokenizer import BaseTokenizer
from aether.tokenizer.utils import fast_split
from aether.tokenizer.byte_level_encoder import ByteLevelEncoder

class BPETokenizer(BaseTokenizer):
    def __init__(self, vocab_size=10000, use_gpu=False, byte_level=False):
        super().__init__(vocab_size)
        self.bpe_ranks = {}
        self.use_gpu = use_gpu
        self.byte_level = byte_level
        if self.byte_level:
            self.ble = ByteLevelEncoder()

    def get_stats(self, corpus):
        pairs = defaultdict(int)
        for word, freq in corpus.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pairs[(symbols[i], symbols[i+1])] += freq
        return pairs

    def merge_vocab(self, pair, corpus):
        pattern = re.escape(' '.join(pair))
        pattern = re.compile(r'(?<!\S)' + pattern + r'(?!\S)')
        new_corpus = {}
        for word in corpus:
            new_word = pattern.sub(''.join(pair), word)
            new_corpus[new_word] = corpus[word]
        return new_corpus

    def train(self, data):
        if self.byte_level:
            print("Training with Byte-Level BPE encoding... 🧠")
            data = [' '.join(map(str, self.ble.encode(text))) + ' </w>' for text in data]
        else:
            data = [' '.join(word) + ' </w>' for word in fast_split(data)]

        corpus = Counter(data)
        for i in range(self.vocab_size):
            pairs = self.get_stats(corpus)
            if not pairs:
                break
            best = max(pairs, key=pairs.get)
            corpus = self.merge_vocab(best, corpus)
            self.bpe_ranks[best] = i

        self._build_vocab(corpus)

    def _build_vocab(self, corpus):
        tokens = set()
        for word in corpus:
            tokens.update(word.split())
        idx_offset = len(self.token2id)
        for idx, tok in enumerate(sorted(tokens)):
            self.token2id[tok] = idx + idx_offset
            self.id2token[idx + idx_offset] = tok

    def encode(self, text, add_special_tokens=True):
        if self.byte_level:
            tokens = list(map(str, self.ble.encode(text))) + ['</w>']
        else:
            tokens = list(text) + ['</w>']

        while True:
            pairs = [(tokens[i], tokens[i+1]) for i in range(len(tokens)-1)]
            pair_ranks = {pair: self.bpe_ranks.get(pair, float('inf')) for pair in pairs}
            if not pair_ranks:
                break
            best_pair = min(pair_ranks, key=pair_ranks.get)
            if pair_ranks[best_pair] == float('inf'):
                break
            first, second = best_pair
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i < len(tokens)-1 and tokens[i] == first and tokens[i+1] == second:
                    new_tokens.append(first + second)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens

        token_ids = [self.token2id.get(tok, self.token2id.get("[UNK]", 1)) for tok in tokens]
        if add_special_tokens:
            token_ids = [self.token2id.get("[BOS]", 2)] + token_ids + [self.token2id.get("[EOS]", 3)]
        return token_ids

    def decode(self, ids, skip_special_tokens=True):
        tokens = [self.id2token[i] for i in ids]
        if skip_special_tokens:
            tokens = [t for t in tokens if t not in {"[PAD]", "[UNK]", "[BOS]", "[EOS]"}]

        if self.byte_level:
            decoded_bytes = [int(tok) for tok in tokens if tok != '</w>']
            return self.ble.decode(decoded_bytes)
        else:
            return ''.join(tokens).replace('</w>', '')