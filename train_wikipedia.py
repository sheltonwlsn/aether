from datasets import load_dataset
import torch
from torch.nn.utils.rnn import pad_sequence
from aether.aether_tokenizer import AetherTokenizer
from streaming_dataset_from_hf import StreamingDatasetFromHF
from transformers import Trainer, TrainingArguments, AutoModelForMaskedLM

# 1. Load Wikipedia (streaming mode)
dataset = load_dataset("wikipedia", "20220301.en", split="train", streaming=True)

# 2. Load Aether tokenizer
tokenizer = AetherTokenizer("tokenizer.json")

# 3. Wrap streaming dataset
streaming_dataset = StreamingDatasetFromHF(
    hf_streaming_dataset=dataset,
    tokenizer=tokenizer,
    max_tokens_per_batch=2048
)

# 4. Load small model for now
model = AutoModelForMaskedLM.from_pretrained("distilbert-base-uncased")

def aether_collator(features, mlm_probability=0.15):
    input_ids = []
    attention_masks = []

    for batch in features:
        input_ids.extend(batch["input_ids"])
        attention_masks.extend(batch["attention_mask"])

    input_ids = [torch.tensor(x, dtype=torch.long) for x in input_ids]
    attention_masks = [torch.tensor(x, dtype=torch.long) for x in attention_masks]

    # Truncate to model's max length
    input_ids = [x[:512] for x in input_ids]
    attention_masks = [x[:512] for x in attention_masks]

    # Pad
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    attention_masks = pad_sequence(attention_masks, batch_first=True, padding_value=0)

    # Create labels
    labels = input_ids.clone()

    # Randomly mask tokens (standard 15% Masked LM probability)
    probability_matrix = torch.full(labels.shape, mlm_probability)
    special_tokens_mask = labels == 0  # don't mask padding
    probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
    masked_indices = torch.bernoulli(probability_matrix).bool()

    labels[~masked_indices] = -100  # ignore non-masked tokens in loss

    return {
        "input_ids": input_ids,
        "attention_mask": attention_masks,
        "labels": labels,
    }


# 5. Set training arguments
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,  # optional if you want bigger effective batch
    logging_dir="./logs",
    max_steps=1000,  # limit for testing
    logging_steps=50,
    save_steps=500,
    report_to="none"  # disable wandb if you don't use it
)

# 6. Create Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=streaming_dataset,
    data_collator=aether_collator,
)

# 7. Launch training
trainer.train()
