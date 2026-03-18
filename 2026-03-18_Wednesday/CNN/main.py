import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow.keras import layers, models

class SimpleCNN:
    def __init__(self):
        self.model = None
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None
        self.history = None

    def load_data(self):
        """
        This method is used to load the MNIST data
        """
        try:
            (self.x_train, self.y_train), (self.x_test, self.y_test) = tf.keras.datasets.mnist.load_data()

            # Normalize pixel values (0-255 -> 0-1)
            self.x_train = self.x_train / 255.0
            self.x_test = self.x_test / 255.0

            # Add channel dimension for CNN
            self.x_train = self.x_train.reshape(-1, 28, 28, 1)
            self.x_test = self.x_test.reshape(-1, 28, 28, 1)

            print("Train Shape:", self.x_train.shape)
            print("Test Shape:", self.x_test.shape)
        except Exception as e:
            print("Error in loading MNIST data",e)

    def build_model(self):
        """
        This method is used to build the CNN model
        """
        try:
            self.model = models.Sequential([
                layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
                layers.MaxPooling2D((2, 2)),

                layers.Conv2D(64, (3, 3), activation='relu'),
                layers.MaxPooling2D((2, 2)),

                layers.Flatten(),

                layers.Dense(64, activation='relu'),

                layers.Dense(10, activation='softmax')
            ])

            self.model.compile(
                optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
            self.model.summary()
        except Exception as e:
            print("Error in building model",e)

    def train_model(self):
        """
        this method is used to train the CNN model
        """
        try:
            self.history = self.model.fit(
                self.x_train,
                self.y_train,
                epochs=5,
                validation_data=(self.x_test, self.y_test)
            )
        except Exception as e:
            print("Error in training model",e)

    def evaluate_model(self):
        """
        this method is used to evaluate the CNN model
        """
        try:
            loss, accuracy = self.model.evaluate(self.x_test, self.y_test)
            print("Test Accuracy:", accuracy)
        except Exception as e:
            print("Error in evaluating model",e)

    def plot_performance(self):
        """
        this method is used to plot the performance of the CNN model
        """
        try:
            self.evaluate_model()
            plt.figure(figsize=(12, 4))
            # plot for accuracy
            plt.subplot(1, 2, 1)
            plt.plot(self.history.history['accuracy'], label='Train Accuracy')
            plt.plot(self.history.history['val_accuracy'], label='Val Accuracy')
            plt.title('Model Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.legend()
            # plot for loss
            plt.subplot(1, 2, 2)
            plt.plot(self.history.history['loss'], label='Train Loss')
            plt.plot(self.history.history['val_loss'], label='Val Loss')
            plt.title('Model Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()

            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"Error while plotting: {e}")

def main():
    """
    this is the driver function for the CNN model
    """
    cnn = SimpleCNN()
    cnn.load_data()
    cnn.build_model()
    cnn.train_model()
    cnn.evaluate_model()
    cnn.plot_performance()

if __name__ == "__main__":
    main()
