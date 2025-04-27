import time
from aether.tokenizer.sentencepiece_tokenizer import SentencePieceTokenizer

def train_and_benchmark(model_type, vocab_size, training_data):
    tokenizer = SentencePieceTokenizer(
        vocab_size=vocab_size,
        model_type=model_type,
        use_gpu=False
    )

    start_time = time.time()
    tokenizer.train(training_data)
    elapsed = time.time() - start_time

    reconstruction_errors = 0
    total_sentences = len(training_data)

    for sentence in training_data:
        encoded = tokenizer.encode(sentence)
        decoded = tokenizer.decode(encoded)

        # Normalize spaces for comparison
        orig = ' '.join(sentence.lower().split())
        decoded = ' '.join(decoded.lower().split())

        if orig != decoded:
            reconstruction_errors += 1

    reconstruction_accuracy = (total_sentences - reconstruction_errors) / total_sentences

    return elapsed, reconstruction_accuracy

if __name__ == "__main__":
    # Prepare training data
    training_data = [
        "Thanos bends reality itself.",
        "The Aether flows through unseen dimensions.",
        "Quantum knowledge transcends singularity.",
        "Magic intertwines with the cosmic web.",
        "The fabric of reality unfolds beyond dimensions.",
        "Magic is a manifestation of unseen forces.",
        "Gravity bends the fabric of space-time.",
        "Knowledge expands through universal networks.",
        "Illusion shatters human perception.",
        "Energy flows through complex dimensions."
    ]

    vocab_size = 8000

    print("Benchmarking Unigram Model...")
    unigram_time, unigram_accuracy = train_and_benchmark("unigram", vocab_size, training_data)
    print(f"⏱ Unigram Training Time: {unigram_time:.2f}s")
    print(f"✅ Unigram Reconstruction Accuracy: {unigram_accuracy:.2%}")

    print("\nBenchmarking BPE Model...")
    bpe_time, bpe_accuracy = train_and_benchmark("bpe", vocab_size, training_data)
    print(f"⏱ BPE Training Time: {bpe_time:.2f}s")
    print(f"✅ BPE Reconstruction Accuracy: {bpe_accuracy:.2%}")
