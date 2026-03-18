import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import LSTM, Dense, Embedding

class SentimentAnalysisLSTM:
    def __init__(self):
        self.data = None
        self.sentiments = None
        self.tokenizer = Tokenizer()
        self.tokenized_data = None
        self.max_len = 5
        self.model= None

    def load_data(self):
        try:
            data = {
            'Review' : [
                'I love this product',
                'This is amazing',
                'Very bad experience',
                'I hate this item',
                'Excellent quality',
                'Worst purchase ever',
                'Really happy with this',
                'Not good at all',
                'Superb performance',
                'Terrible service'
            ],
            'Sentiment' : [1,1,0,0,1,0,1,0,1,0]
            }
            self.data = pd.DataFrame(data)
        except Exception as e:
            print(f'Error in loading data : {e}')

    def preprocess_text(self):
        try:
            texts = self.data['Review']
            self.sentiments = self.data['Sentiment']
            # Converting words into numbers using Tokenizer
            # example: I hate this item = [1,2,3,4]
            self.tokenizer.fit_on_texts(texts)
            # Convert sentences into numeric sequence
            sequence = self.tokenizer.texts_to_sequences(texts)
            print(f'Sequence:{sequence}')
            # if maxlength=5 then in above sentence it will add padding and it become[0,1,2,3,4]
            self.tokenized_data = pad_sequences(sequence,maxlen=self.max_len)
            print(f'Tokenized Data:\n{self.tokenized_data}')
        except Exception as e:
            print(f'Error in preprocessing text : {e}')

    def build_model(self):
        try:
            vocab_size =len(self.tokenizer.word_index)+1
            # we have done +1 because index start from 1 and zero reserved for padding
            print(vocab_size)
            self.model = Sequential()
            self.model.add(Embedding(
                input_dim = vocab_size,
                output_dim = 8,
                input_length = self.max_len
            ))
            self.model.add(LSTM(16))
            self.model.add(Dense(1,activation='sigmoid'))
            self.model.compile(
                    optimizer='adam',
                    loss='binary_crossentropy',
                    metrics=['accuracy']
            )
            print('Model Build successfully')
        except Exception as e:
            print(f'Error in building model : {e}')

    def train_model(self):
        try:
            self.model.fit(
                self.tokenized_data,
                self.sentiments,
                epochs = 20,
                verbose = 1
            )
        except Exception as e:
            print(f'Error in training model : {e}')

    def predict(self):
        try:
            test_text = ['service was too good']
            seq = self.tokenizer.texts_to_sequences(test_text)
            padded = pad_sequences(seq, maxlen=self.max_len)

            prediction = self.model.predict(padded)
            print(f'Test Input:{test_text}')
            print(f"Prediction:{prediction}")

            if prediction > 0.5:
                print("Positive Sentiment")
            else:
                print('Negative Sentiment')
        except Exception as e:
            print(f'Error in predicting : {e}')

def main():
    lstm = SentimentAnalysisLSTM()
    lstm.load_data()
    lstm.preprocess_text()
    lstm.build_model()
    lstm.train_model()
    lstm.predict()

if __name__ == '__main__':
    main()