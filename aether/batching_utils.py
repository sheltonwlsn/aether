def count_tokens(tokenizer, text):
    """Count number of tokens for a single text."""
    return len(tokenizer.encode(text, add_special_tokens=False))

def batch_texts(tokenizer, texts, max_tokens_per_batch=2048):
    """
    Group texts into batches without exceeding max tokens per batch.
    
    Args:
        tokenizer: Tokenizer object
        texts: List of strings
        max_tokens_per_batch: Max total tokens per batch
        
    Returns:
        List of batches (each batch is a list of texts).
    """
    batches = []
    current_batch = []
    current_tokens = 0

    for text in texts:
        num_tokens = count_tokens(tokenizer, text)

        # If adding this text would overflow the max tokens, start a new batch
        if current_tokens + num_tokens > max_tokens_per_batch:
            if current_batch:
                batches.append(current_batch)
            current_batch = [text]
            current_tokens = num_tokens
        else:
            current_batch.append(text)
            current_tokens += num_tokens

    # Add the last batch if it's non-empty
    if current_batch:
        batches.append(current_batch)

    return batches