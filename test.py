from aether.tokenizer.bpe_tokenizer import BPETokenizer

# Step 1: Load your trained tokenizer
tokenizer = BPETokenizer(byte_level=True)
tokenizer.load("tokenizer.json")

# Step 2: Text you want to analyze
text = "The multiverse bends reality itself."

# Step 3: Encode without special tokens
ids = tokenizer.encode(text, add_special_tokens=False)

# Step 4: Map IDs back to tokens
tokens = [tokenizer.id2token[i] for i in ids]

# Step 5: Pretty print
print(f"\nOriginal Text: {text}\n")
print(f"{'Token':<15} {'ID'}")
print("-" * 30)
for token, id_ in zip(tokens, ids):
    print(f"{token:<15} {id_}")

# Step 6: Token count
print("\n" + "-" * 30)
print(f"Total Tokens: {len(tokens)}")
