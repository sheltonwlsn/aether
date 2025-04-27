import time
import json
from datasets import load_dataset
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

        orig = ' '.join(sentence.lower().split())
        decoded = ' '.join(decoded.lower().split())

        if orig != decoded:
            reconstruction_errors += 1

    reconstruction_accuracy = (total_sentences - reconstruction_errors) / total_sentences

    return elapsed, reconstruction_accuracy

if __name__ == "__main__":
    # Step 1: Stream Wikipedia
    wiki = load_dataset("wikipedia", "20220301.en", split="train", streaming=True)
    N = 50000
    raw_texts = []
    for example in wiki:
        text = example["text"].strip()
        if text:
            raw_texts.append(text)
        if len(raw_texts) >= N:
            break
    print(f"Collected {len(raw_texts)} Wikipedia documents.")

    # Step 2: Split into sentences and limit length
    texts = []
    for paragraph in raw_texts:
        sentences = paragraph.split('.')
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence and len(sentence) < 200:
                texts.append(sentence)

    print(f"Prepared {len(texts)} Wikipedia sentences after splitting and filtering.")

    # Step 3: Benchmark
    vocab_size = 8000
    results = {}

    print("Benchmarking Unigram Model...")
    unigram_time, unigram_accuracy = train_and_benchmark("unigram", vocab_size, texts)
    results["unigram"] = {
        "training_time_seconds": round(unigram_time, 2),
        "reconstruction_accuracy_percent": round(unigram_accuracy * 100, 2)
    }
    print(f"⏱ Unigram Training Time: {unigram_time:.2f}s")
    print(f"✅ Unigram Reconstruction Accuracy: {unigram_accuracy:.2%}")

    print("\nBenchmarking BPE Model...")
    bpe_time, bpe_accuracy = train_and_benchmark("bpe", vocab_size, texts)
    results["bpe"] = {
        "training_time_seconds": round(bpe_time, 2),
        "reconstruction_accuracy_percent": round(bpe_accuracy * 100, 2)
    }
    print(f"⏱ BPE Training Time: {bpe_time:.2f}s")
    print(f"✅ BPE Reconstruction Accuracy: {bpe_accuracy:.2%}")

    # Step 4: Save results to JSON
    with open("benchmark_results_wikipedia.json", "w") as f:
        json.dump(results, f, indent=4)

    print("\nBenchmark results saved to benchmark_results_wikipedia.json!")