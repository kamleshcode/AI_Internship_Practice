import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

separator = "\n" + "--" * 40 + "\n"

class TitanicDeepLearning:
    def __init__(self, filepath):
        self.filepath = filepath
        self.df = None
        self.model = None
        self.preprocessor = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.history = None

    def load_dataset(self):
        try:
            self.df = pd.read_csv(self.filepath)
            print("Dataset Loaded")
            print(self.df.head(), end=separator)
        except FileNotFoundError as e:
            print(f'File Not Found: {e}')

    def preprocess_data(self):
        try:
            print("Data Info")
            print(self.df.info(), end=separator)

            # Drop unnecessary columns
            self.df.drop(columns=["Name", "Ticket", "PassengerId", "Cabin"], inplace=True)

            # Fill missing values
            self.df["Age"] = self.df["Age"].fillna(self.df["Age"].median())
            self.df["Embarked"] = self.df["Embarked"].fillna(self.df["Embarked"].mode()[0])

            # Feature Engineering
            self.df["FamilySize"] = self.df["SibSp"] + self.df["Parch"] + 1
            self.df.drop(columns=["SibSp", "Parch"], inplace=True)

            print("Preprocessing Done")
            print(self.df.head(), end=separator)
        except Exception as e:
            print(f'Error in preprocessing data: {e}')

    def visualize(self):
        try:
            plt.figure(figsize=(10, 8))
            corr = self.df.corr(numeric_only=True)
            sns.heatmap(corr, annot=True, cmap='coolwarm')
            plt.title("Feature Correlation Heatmap")
            plt.show()
        except Exception as e:
            print(f'Error in visualizing data: {e}')

    def split_data(self):
        self.X = self.df.drop(columns="Survived")
        self.y = self.df["Survived"]

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X,self.y, test_size=0.2, random_state=42)

    def build_pipeline(self):
        numeric_features = ["Age","Fare","Pclass","FamilySize"]
        category_features = ["Embarked","Sex"]

        numeric_pipeline = Pipeline([
            ("scaler", StandardScaler())
        ])

        categorical_pipeline = Pipeline([
            ("encoder",OneHotEncoder(handle_unknown="ignore"))
        ])

        self.preprocessor = ColumnTransformer([
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, category_features),
        ])
        self.X_train = self.preprocessor.fit_transform(self.X_train)
        self.X_test = self.preprocessor.transform(self.X_test)


    def build_model(self):
        input_shape=self.X_train.shape[1]
        self.model = Sequential([
            Dense(32, activation='relu', input_shape=(input_shape,)),
            Dense(16, activation='relu'),
            Dense(1, activation='sigmoid')
        ])

        self.model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        print("Model Built")
        self.model.summary()

    def train_model(self):
        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        )
        self.history = self.model.fit(
            self.X_train,
            self.y_train,
            batch_size=16,
            epochs=50,
            validation_data=(self.X_test, self.y_test),
            callbacks=[early_stop]
        )

    def evaluate_model(self):
        predictions = self.model.predict(self.X_test)
        predictions = (predictions > 0.5).astype(int)

        print("\nAccuracy Score:")
        print(accuracy_score(self.y_test, predictions))
        print("\nConfusion Matrix:")
        print(confusion_matrix(self.y_test, predictions))
        print("\nClassification Report:")
        print(classification_report(self.y_test, predictions))

    def plot_history(self):
        plt.figure(figsize=(8,5))
        plt.plot(self.history.history['loss'],label='Training Loss')
        plt.plot(self.history.history['val_loss'],label='Validation Loss')
        plt.title('Loss Curve')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend()
        plt.show()


if __name__ == "__main__":
    obj = TitanicDeepLearning("Titanic-Dataset.csv")
    obj.load_dataset()
    obj.preprocess_data()
    obj.visualize()
    obj.split_data()
    obj.build_pipeline()
    obj.build_model()
    obj.train_model()
    obj.evaluate_model()
    obj.plot_history()