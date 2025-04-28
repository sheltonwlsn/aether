# 🚀 Aether Tokenizer Benchmark Report

This document summarizes benchmarking results comparing **Unigram** and **BPE** models trained using the Aether SentencePiece-style tokenizer implementation.

---

## 📚 Dataset

- 10 sentences (custom small corpus)
- English language
- Pre-tokenized using Aether `fast_split`

---

## ⚙️ Training Configuration

- Vocabulary size: 8000
- Dropout: 0.0
- Use GPU: False (M1 Pro CPU)
- Training runs: 1 epoch (no multi-cycle tuning)

---

## 📊 Results

| Metric | Unigram | BPE |
|:---|:---|:---|
| Training Time | ⏱ 0.06 seconds | ⏱ 0.01 seconds |
| Reconstruction Accuracy | ✅ 100.00% | ✅ 100.00% |

---

## 🧠 Key Observations

- **Speed:** BPE trains ~6x faster than Unigram on this small dataset.
- **Quality:** Both models achieve perfect 100% reconstruction accuracy.
- **Efficiency:** Optimized substring counting dramatically speeds up Unigram training.

---

## 🎯 Future Benchmarks (Planned)

- Scale to 10,000+ sentences
- Vary vocabulary sizes (8k, 16k, 32k, 64k)
- Benchmark on streaming Wikipedia datasets
- GPU-accelerated tokenizers (future)

---

# 🌌 Aether: Built for real-world LLM pretraining.