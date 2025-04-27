import torch
from torch.utils.data import IterableDataset

class StreamingDatasetFromHF(IterableDataset):
    def __init__(self, hf_streaming_dataset, tokenizer, max_tokens_per_batch=2048):
        super().__init__()
        self.dataset = hf_streaming_dataset
        self.tokenizer = tokenizer
        self.max_tokens_per_batch = max_tokens_per_batch

    def __iter__(self):
        batch_input_ids = []
        batch_attention_masks = []
        current_tokens = 0

        for example in self.dataset:
            line = example.get("text", "").strip()
            if not line:
                continue

            encoded = self.tokenizer.encode_plus(line)
            input_ids = encoded["input_ids"]
            attention_mask = encoded["attention_mask"]

            if current_tokens + len(input_ids) > self.max_tokens_per_batch:
                if batch_input_ids:
                    yield {
                        "input_ids": torch.nn.utils.rnn.pad_sequence(
                            [torch.tensor(ids, dtype=torch.long) for ids in batch_input_ids],
                            batch_first=True,
                            padding_value=0
                        ),
                        "attention_mask": torch.nn.utils.rnn.pad_sequence(
                            [torch.tensor(masks, dtype=torch.long) for masks in batch_attention_masks],
                            batch_first=True,
                            padding_value=0
                        ),
                    }
                batch_input_ids = []
                batch_attention_masks = []
                current_tokens = 0

            batch_input_ids.append(input_ids)
            batch_attention_masks.append(attention_mask)
            current_tokens += len(input_ids)

        if batch_input_ids:
            yield {
                "input_ids": torch.nn.utils.rnn.pad_sequence(
                    [torch.tensor(ids, dtype=torch.long) for ids in batch_input_ids],
                    batch_first=True,
                    padding_value=0
                ),
                "attention_mask": torch.nn.utils.rnn.pad_sequence(
                    [torch.tensor(masks, dtype=torch.long) for masks in batch_attention_masks],
                    batch_first=True,
                    padding_value=0
                ),
            }