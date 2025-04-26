import torch

def create_initial_batch(texts, char2id, max_len=128):
    """
    Create padded tensor batch from raw texts.
    """
    batch = []
    for text in texts:
        ids = [char2id.get(c, 0) for c in text]
        if len(ids) < max_len:
            ids += [0] * (max_len - len(ids))
        else:
            ids = ids[:max_len]
        batch.append(ids)
    batch_tensor = torch.tensor(batch, dtype=torch.long)
    return batch_tensor

def find_most_frequent_pairs(batch_tensor):
    """
    Find the most frequent token pair across batch.
    """
    a = batch_tensor[:, :-1]
    b = batch_tensor[:, 1:]
    valid = (a > 0) & (b > 0)
    pairs = a[valid] * 10000 + b[valid]
    if pairs.numel() == 0:
        return None
    counts = torch.bincount(pairs)
    if counts.numel() == 0:
        return None
    max_idx = counts.argmax()
    token1 = max_idx // 10000
    token2 = max_idx % 10000
    return (token1.item(), token2.item())

def merge_batch(batch_tensor, pair_to_merge, new_token_id):
    """
    Merge all occurrences of a given pair in the batch.
    """
    a, b = pair_to_merge
    mask = (batch_tensor[:, :-1] == a) & (batch_tensor[:, 1:] == b)
    new_batch = []
    for i in range(batch_tensor.size(0)):
        tokens = []
        skip = False
        for j in range(batch_tensor.size(1)):
            if skip:
                skip = False
                continue
            if j < batch_tensor.size(1) - 1 and mask[i, j]:
                tokens.append(new_token_id)
                skip = True
            else:
                tokens.append(batch_tensor[i, j].item())
        tokens = tokens + [0] * (batch_tensor.size(1) - len(tokens))
        new_batch.append(tokens)
    return torch.tensor(new_batch, dtype=torch.long)