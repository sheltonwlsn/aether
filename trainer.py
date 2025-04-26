import time
import argparse
from tokenizer.bpe_tokenizer import BPETokenizer
from tokenizer.unigram_tokenizer import UnigramTokenizer
from tokenizer.wordpiece_tokenizer import WordPieceTokenizer
from tokenizer.sentencepiece_tokenizer import SentencePieceTokenizer

def load_corpus_in_batches(path, batch_size=50000):
    batch = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                batch.append(line)
            if len(batch) >= batch_size:
                yield batch
                batch = []
        if batch:
            yield batch

def benchmark(corpus_path, model_name, vocab_size, use_gpu, dynamic_batch_size):
    if model_name == "bpe":
        tokenizer = BPETokenizer(vocab_size=vocab_size, use_gpu=use_gpu)
    elif model_name == "unigram":
        tokenizer = UnigramTokenizer(vocab_size=vocab_size, use_gpu=use_gpu)
    elif model_name == "wordpiece":
        tokenizer = WordPieceTokenizer(vocab_size=vocab_size, use_gpu=use_gpu)
    elif model_name == "sentencepiece_bpe":
        tokenizer = SentencePieceTokenizer(vocab_size=vocab_size, model_type="bpe", use_gpu=use_gpu)
    elif model_name == "sentencepiece_unigram":
        tokenizer = SentencePieceTokenizer(vocab_size=vocab_size, model_type="unigram", use_gpu=use_gpu)
    else:
        raise ValueError(f"Unknown model {model_name}")

    start = time.time()

    if dynamic_batch_size:
        print(f"Training dynamically with batch size {dynamic_batch_size}...")
        for batch in load_corpus_in_batches(corpus_path, dynamic_batch_size):
            tokenizer.train(batch)
    else:
        with open(corpus_path, 'r', encoding='utf-8') as f:
            data = f.read().splitlines()
        tokenizer.train(data)

    end = time.time()
    print(f"Training time for {model_name} (GPU={use_gpu}): {end - start:.2f} seconds")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--corpus', type=str, required=True)
    parser.add_argument('--model', type=str, default="bpe", choices=["bpe", "unigram", "wordpiece", "sentencepiece_bpe", "sentencepiece_unigram"])
    parser.add_argument('--vocab_size', type=int, default=8000)
    parser.add_argument('--use_gpu', action='store_true')
    parser.add_argument('--dynamic_batch_size', type=int, default=0)
    args = parser.parse_args()

    benchmark(args.corpus, args.model, args.vocab_size, args.use_gpu, args.dynamic_batch_size)

if __name__ == "__main__":
    main()