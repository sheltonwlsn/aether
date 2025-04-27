from datasets import Dataset
import torch
from transformers import Trainer, TrainingArguments, DataCollatorWithPadding, AutoModelForSequenceClassification
from aether.aether_tokenizer import AetherTokenizer

# Step 1: Load your Aether tokenizer
tokenizer = AetherTokenizer("tokenizer.json")

# Step 2: Prepare a sample dataset
texts = [
    {"text": "The multiverse expands beyond imagination."},
    {"text": "Aether flows through unseen dimensions."},
    {"text": "Quantum knowledge transcends singularity."},
    {"text": "Magic intertwines with the cosmic web."},
    {"text": "Reality is a fragile illusion."}
]

dataset = Dataset.from_list(texts)

# Step 3: Tokenize the dataset
def tokenize_function(examples):
    return tokenizer.encode_plus(examples["text"])

tokenized_dataset = dataset.map(tokenize_function, batched=False)

# Step 4: Create a DataCollator
# Since AetherTokenizer is not a HuggingFace PreTrainedTokenizerFast,
# we define a simple wrapper to simulate padding (manual padding can be added later if needed).
class SimpleDataCollator:
    def __call__(self, features):
        batch_input_ids = [f["input_ids"] for f in features]
        batch_attention_mask = [f["attention_mask"] for f in features]

        # Find max length
        max_length = max(len(ids) for ids in batch_input_ids)

        # Pad manually
        padded_input_ids = [ids + [0]*(max_length - len(ids)) for ids in batch_input_ids]
        padded_attention_mask = [mask + [0]*(max_length - len(mask)) for mask in batch_attention_mask]

        # 🔥 Convert to torch.Tensor
        return {
            "input_ids": torch.tensor(padded_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(padded_attention_mask, dtype=torch.long),
            "labels": torch.zeros(len(features), dtype=torch.long)  # Dummy labels
        }

data_collator = SimpleDataCollator()

# Step 5: Load a tiny model
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

# Step 6: Trainer setup
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=2,
    num_train_epochs=1,
    logging_dir="./logs",
    logging_steps=1,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=None,  # AetherTokenizer doesn't implement full PreTrainedTokenizerFast yet
    data_collator=data_collator,
)

# Step 7: Train
trainer.train()