import tensorflow as tf
from tensorflow.keras import layers
import numpy as np

class Tokenizer:
    """Map words to IDs and back."""
    def __init__(self, texts):
        self.words = sorted(set(" ".join(texts).split()))
        self.w2i = {w: i for i, w in enumerate(self.words)}
        self.i2w = {i: w for w, i in self.w2i.items()}

    def encode(self, text):
        return [self.w2i[w] for w in text.split() if w in self.w2i]

    def decode(self, ids):
        return " ".join([self.i2w[int(i)] for i in ids])


def main():
    """Train on a story and generate text."""
    story = ["the robot found a key", "the key opened a door", "behind the door was magic"]
    tk = Tokenizer(story)
    

if __name__ == "__main__":
    main()
