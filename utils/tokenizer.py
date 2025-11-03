class DummyTokenizer:
    def __init__(self, vocab_size: int = 32000):
        self.vocab_size = vocab_size

    def encode(self, text: str):
        return [ord(c) % self.vocab_size for c in text]

    def decode(self, ids):
        return "".join(chr(i) for i in ids)
