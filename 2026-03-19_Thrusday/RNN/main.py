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

    def build_model(self):
        try:
            vocab_size = len(self.tokenizer.word_index) + 1

            self.model = models.Sequential([
                # input_dim: Size of vocabulary
                # output_dim: Dimension of the dense embedding
                # input_length: Length of input sequences (5)
                layers.Embedding(input_dim=vocab_size, output_dim=8, input_length=self.max_len),

                # SimpleRNN layer with 16 units
                layers.SimpleRNN(16),

                # Final output layer for binary classification (0 or 1)
                layers.Dense(1, activation='sigmoid')
            ])

            self.model.compile(
                optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy']
            )
            print("\nModel Summary :")
            self.model.summary()
            print("RNN Model built successfully.")
        except Exception as e:
            print(f"Error in building model: {e}")

    def train_model(self):
        try:
            self.model.fit(
                self.tokenized_data,
                self.labels,
                epochs=20,
            )
            print("Training complete.")
        except Exception as e:
            print(f"Error in training: {e}")

    def predict_sentiment(self, test_sentence):
        try:
            seq = self.tokenizer.texts_to_sequences([test_sentence])
            padded = pad_sequences(seq, maxlen=self.max_len)
            prediction = self.model.predict(padded, verbose=0)

            sentiment = "Positive" if prediction > 0.5 else "Negative"
            print(f"Sentence: '{test_sentence}'")
            print(f"Confidence: {prediction[0][0]:.4f} -> {sentiment}")
        except Exception as e:
            print(f"Error in prediction: {e}")

def main():
    rnn = SentimentAnalysisRNN()
    rnn.load_data()
    rnn.preprocess_text()
    rnn.build_model()
    rnn.train_model()

    print('\nTesting model :')
    rnn.predict_sentiment("the film was amazing")
    rnn.predict_sentiment("i hate movie")


if __name__ == "__main__":
    main()
