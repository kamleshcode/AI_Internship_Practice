import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import layers, models

class SentimentAnalysisRNN:
    def __init__(self):
        self.data = None
        self.labels = None
        self.tokenizer = Tokenizer()
        self.tokenized_data = None
        self.max_len = 5
        self.model = None

    def load_data(self):
        try:
            sentences = [
                "movie was good",
                "movie was bad",
                "i like this film",
                "i hate this film",
                "this movie is amazing",
                "this movie is terrible",
                "film was nice",
                "film was boring",
                "good acting",
                "bad acting"
            ]
            # 1 = Positive, 0 = Negative
            self.labels = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
            self.data = sentences
            print("Data loaded successfully.")
        except Exception as e:
            print(f"Error in loading data: {e}")

    def preprocess_text(self):
        try:
            # Fit tokenizer on our specific sentences
            self.tokenizer.fit_on_texts(self.data)
            sequences = self.tokenizer.texts_to_sequences(self.data)

            # Pad sequences so they all have a length of 5
            self.tokenized_data = pad_sequences(sequences, maxlen=self.max_len)

            print(f"Vocabulary Size: {len(self.tokenizer.word_index)}")
            print(f'Word Index:{self.tokenizer.word_index}')
            print(f"Tokenized Data:\n{self.tokenized_data}")
        except Exception as e:
            print(f"Error in preprocessing: {e}")


def main():
    rnn = SentimentAnalysisRNN()
    rnn.load_data()
    rnn.preprocess_text()

if __name__ == "__main__":
    main()
