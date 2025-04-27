# Aether Tokenizer 🌌✨

Aether is a production-grade Byte-Level BPE tokenizer, compatible with HuggingFace Trainer, built for scaling from small models to billion-token LLMs.

## Features

- ✅ Byte-Level BPE (like GPT-2/3/4)
- ✅ HuggingFace-compatible tokenizer interface
- ✅ Dynamic token batching
- ✅ Streaming massive datasets (100GB+)
- ✅ Trainer integration ready
- ✅ Easy to extend and customize

---

## Installation

```bash
pip install .
```

---

## Train a Tokenizer

```bash
aether-cli --corpus large_corpus.txt --model bpe --vocab_size 50000 --use_gpu --byte_level
```

Generates a `tokenizer.json`.

---

## Load the Tokenizer

```python
from aether.aether_tokenizer import AetherTokenizer

tokenizer = AetherTokenizer("tokenizer.json")
```

---

## Stream Large Datasets

For local files:

```python
from streaming_dataset import StreamingTextDataset

dataset = StreamingTextDataset(
    file_path="large_corpus.txt",
    tokenizer=tokenizer,
    max_tokens_per_batch=2048
)
```

For HuggingFace datasets (Wikipedia):

```python
from streaming_dataset_from_hf import StreamingDatasetFromHF
from datasets import load_dataset

hf_dataset = load_dataset("wikipedia", "20220301.en", split="train", streaming=True)

dataset = StreamingDatasetFromHF(
    hf_streaming_dataset=hf_dataset,
    tokenizer=tokenizer,
    max_tokens_per_batch=2048
)
```

---

## Training with HuggingFace Trainer

```python
from transformers import Trainer, TrainingArguments, AutoModelForMaskedLM

model = AutoModelForMaskedLM.from_pretrained("distilbert-base-uncased")

training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    max_steps=1000,
    logging_dir="./logs",
    logging_steps=50,
    save_steps=500,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=aether_collator,
)

trainer.train()
```

---

## Training Results ✅

| Metric | Value |
|:---|:---|
| Dataset | Wikipedia (streamed) |
| Model | DistilBERT (Masked Language Modeling) |
| Steps | 1000 |
| Final Loss | ~0.29 |
| Samples/sec | ~8 |
| Steps/sec | ~1 |

✅ Successfully trained on Mac M1 (MPS backend)!

---

## Checkpoints and Model Saving

- Model checkpoints are saved automatically to `./results/checkpoint-XXXX/`
- **Large files are NOT committed to GitHub.**
- Recommended to upload to HuggingFace Hub or external storage for sharing.

Load model:

```python
from transformers import AutoModelForMaskedLM

model = AutoModelForMaskedLM.from_pretrained("./results/checkpoint-1000")
```

---

## Project Structure

```
aether/
├── aether/
│   ├── tokenizer/
│   ├── aether_tokenizer.py
│   ├── streaming_dataset.py
│   ├── streaming_dataset_from_hf.py
├── setup.py
├── README.md
├── tokenizer.json
```

---

## Coming Soon 🚀

- SentencePiece-style sampling
- Aether Transformer (custom LLM)
- Hosted pretraining pipelines
- HuggingFace Hub integration