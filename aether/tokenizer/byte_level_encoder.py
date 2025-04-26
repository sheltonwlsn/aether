class ByteLevelEncoder:
    def __init__(self, add_prefix_space=True):
        self.add_prefix_space = add_prefix_space

    def encode(self, text):
        if self.add_prefix_space and not text.startswith(' '):
            text = ' ' + text
        byte_encoded = list(text.encode('utf-8'))
        return byte_encoded

    def decode(self, ids):
        byte_array = bytes(ids)
        return byte_array.decode('utf-8', errors='replace')