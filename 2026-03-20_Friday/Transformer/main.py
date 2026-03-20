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


class TransformerBlock(layers.Layer):
    """Calculates Attention Scores and Context Vectors."""

    def __init__(self, d, h):
        super().__init__()
        # h = heads, d = model dimension
        self.mha = layers.MultiHeadAttention(num_heads=h, key_dim=d)
        self.norm = layers.LayerNormalization()
        self.ffn = tf.keras.Sequential([layers.Dense(d * 2, activation='relu'), layers.Dense(d)])

    def call(self, x):
        # Create Mask to block future words
        sz = tf.shape(x)[1]
        mask = tf.linalg.band_part(tf.ones((sz, sz)), -1, 0) == 1

        # MHA: Q*K gives Attention Scores -> Softmax gives Weights -> Weights * V gives Context
        context = self.mha(query=x, key=x, value=x, attention_mask=mask[tf.newaxis, tf.newaxis, :, :])

        # Residual + Norm
        x = self.norm(x + context)
        return self.norm(x + self.ffn(x))

def main():
    """Train on a story and generate text."""
    story = ["the robot found a key", "the key opened a door", "behind the door was magic"]
    tk = Tokenizer(story)

    # Prepare Data
    ids = []
    for s in story: ids.extend(tk.encode(s))

    # Input X (3 words) -> Target Y (next 3 words)
    X = np.array([ids[i:i + 3] for i in range(len(ids) - 3)])
    Y = np.array([ids[i + 1:i + 4] for i in range(len(ids) - 3)])


if __name__ == "__main__":
    main()
