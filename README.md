# Aether

**Aether** is a powerful, production-ready tokenizer library for building LLMs and AI models.  
It supports **Byte-Level BPE**, **Unigram LM**, **WordPiece**, **SentencePiece models**, **dynamic batching**, and **GPU acceleration** — all inspired by leading LLM tokenization strategies.

> ✨ The first piece to assembliing the gauntlet (I'm a Marvel guy, not sorry)

---

## 🚀 Features

- 🧠 Byte-Level BPE (GPT-2/3/4 style)
- 🔥 Dynamic corpus batching (train on TBs of text)
- 🚀 GPU-accelerated token merging (optional)
- 🧩 Special tokens: `[PAD]`, `[UNK]`, `[BOS]`, `[EOS]`
- 🎛 Configurable via CLI or Python API
- 📦 Pip-installable package
- 🌐 Unicode-safe, multilingual tokenization
- 🛠 Supports CPU and M1/M2 Mac acceleration

---

## 📦 Installation

```bash
pip install aether
```

(or clone and install locally)

```bash
git clone https://github.com/sheltonwlsn/aether.git
cd aether
pip install .
```

---

## 🔥 Quickstart

**Python API**

```python
from aether.tokenizer.bpe_tokenizer import BPETokenizer

tokenizer = BPETokenizer(vocab_size=8000, byte_level=True)
tokenizer.train(["hello universe 🌌"])

encoded = tokenizer.encode("reality is bending")
print(encoded)

decoded = tokenizer.decode(encoded)
print(decoded)
```

**CLI Training**

```bash
aether-cli --corpus my_corpus.txt --model bpe --vocab_size 8000 --use_gpu
```

---

## 📚 Supported Tokenizers

| Type | Description |
|:---|:---|
| BPE | Classic Byte Pair Encoding |
| Unigram | Unigram Language Model |
| WordPiece | Longest-match subword (like BERT) |
| SentencePiece BPE | Raw-text BPE training |
| SentencePiece Unigram | Raw-text Unigram training |

---

## 🛤 Roadmap

- [ ] Triton GPU accelerated token merging
- [ ] ByteLevel WordPiece
- [ ] Multilingual tokenizer benchmarks
- [ ] Publish to PyPI 🎯

---

## 🤝 License

MIT License

---

> _Reality can be whatever you want it to be._  
> — Shelton WILSON
