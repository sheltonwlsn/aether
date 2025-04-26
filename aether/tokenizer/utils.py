import tiktoken

def fast_split(texts):
    enc = tiktoken.get_encoding("p50k_base")  # fast GPT-3 tokenizer
    if isinstance(texts, str):
        return enc.decode(enc.encode(texts)).split()
    else:
        all_tokens = []
        for text in texts:
            all_tokens.extend(enc.decode(enc.encode(text)).split())
        return all_tokens