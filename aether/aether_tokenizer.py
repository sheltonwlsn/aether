class AetherTokenizer:
    def __init__(self, tokenizer_model_path):
        from aether.tokenizer.bpe_tokenizer import BPETokenizer
        self.tokenizer = BPETokenizer(byte_level=True)
        self.tokenizer.load(tokenizer_model_path)

    def tokenize(self, text):
        """Tokenize text into subwords (no IDs)."""
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        return [self.tokenizer.id2token[i] for i in ids]

    def encode(self, text, add_special_tokens=True):
        """Encode text into token IDs."""
        return self.tokenizer.encode(text, add_special_tokens=add_special_tokens)

    def decode(self, ids, skip_special_tokens=True):
        """Decode token IDs back into text."""
        return self.tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)

    def encode_plus(self, text, add_special_tokens=True):
        """Encode text and return a dictionary like HuggingFace style."""
        ids = self.encode(text, add_special_tokens=add_special_tokens)
        return {
            "input_ids": ids,
            "attention_mask": [1] * len(ids)
        }

    def batch_encode_plus(self, texts, add_special_tokens=True):
        """Batch encode multiple texts."""
        batch_input_ids = []
        batch_attention_masks = []

        for text in texts:
            encoded = self.encode_plus(text, add_special_tokens=add_special_tokens)
            batch_input_ids.append(encoded["input_ids"])
            batch_attention_masks.append(encoded["attention_mask"])

        return {
            "input_ids": batch_input_ids,
            "attention_mask": batch_attention_masks
        }