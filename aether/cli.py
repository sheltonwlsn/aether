import argparse
from aether.tokenizer.bpe_tokenizer import BPETokenizer
from aether.tokenizer.unigram_tokenizer import UnigramTokenizer
from aether.tokenizer.wordpiece_tokenizer import WordPieceTokenizer
from aether.tokenizer.sentencepiece_tokenizer import SentencePieceTokenizer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--corpus', type=str, required=True)
    parser.add_argument('--output', type=str, default="tokenizer.json")
    parser.add_argument('--vocab_size', type=int, default=10000)
    parser.add_argument('--model', type=str, default="bpe", choices=["bpe", "unigram", "wordpiece", "sentencepiece_bpe", "sentencepiece_unigram"])
    parser.add_argument('--use_gpu', action='store_true')
    parser.add_argument('--byte_level', action='store_true')
    args = parser.parse_args()

    with open(args.corpus) as f:
        data = f.read().splitlines()

    if args.model == "bpe":
        tokenizer = BPETokenizer(vocab_size=args.vocab_size, use_gpu=args.use_gpu, byte_level=args.byte_level)
    elif args.model == "unigram":
        tokenizer = UnigramTokenizer(vocab_size=args.vocab_size, use_gpu=args.use_gpu)
    elif args.model == "wordpiece":
        tokenizer = WordPieceTokenizer(vocab_size=args.vocab_size, use_gpu=args.use_gpu)
    elif args.model == "sentencepiece_bpe":
        tokenizer = SentencePieceTokenizer(vocab_size=args.vocab_size, model_type="bpe", use_gpu=args.use_gpu)
    elif args.model == "sentencepiece_unigram":
        tokenizer = SentencePieceTokenizer(vocab_size=args.vocab_size, model_type="unigram", use_gpu=args.use_gpu)
    else:
        raise ValueError(f"Unknown model {args.model}")

    tokenizer.train(data)
    tokenizer.save(args.output)

if __name__ == "__main__":
    main()