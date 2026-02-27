import category_encoders as ce
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sqlalchemy import create_engine
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


class MSSQLDataLoader:
    def __init__(self, server, database, driver="ODBC Driver 17 for SQL Server"):
        """
        Establishing database connection
        """
        try:
            connection_string = (
                f"mssql+pyodbc://{server}/{database}"
                f"?driver={driver.replace(' ', '+')}"
            )
            self.engine = create_engine(connection_string)
        except Exception as e:
            print("Error in connecting to database : ",e )

    def load_table(self, table_name):
        """
        Load full table from SQL Server
        """
        try:
            query = f"SELECT * FROM {table_name}"
            df = pd.read_sql(query, self.engine)
            print("Data loaded successfully")
            return df
        except Exception as e:
            print("Error in loading data ...",e)

class CarEvaluationPipeline:
    def __init__(self, df):
        self.df = df
        self.label_encoders = {}
        self.model = None
        self.X=None
        self.y=None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.encoder = None

    def perform_eda(self):
        """
        This function is used for EDA on df
        """
        try:
            print("Performing EDA ...")
            print("\nDataset Shape:", self.df.shape)
            print("\nDataset Info")
            print(self.df.info())
            print("\nClass Distribution")
            print(self.df["class"].value_counts())
            sns.countplot(x="class", data=self.df)
            plt.title("Class Distribution")
            plt.show()
            print("Class Distribution Completed")
        except Exception as e:
            print("Error in performing eda ...",e)

    def preprocess(self,df):
        """
        this function is used for preprocessing df
        """
        try:
            print("Preprocessing ...")
            df['persons'] = df['persons'].fillna(5)
            df['doors'] = df['doors'].replace("5more", 5)
            df["doors"] = df["doors"].fillna(5)
            df["persons"] = df["persons"].astype("int64")
            df["doors"] = df["doors"].astype("int64")
            self.X = df.drop("class", axis=1)
            self.y = df["class"]
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
            self.encoder = ce.OrdinalEncoder(cols=['buying', 'maint', 'lug_boot', 'safety'])
            self.X_train = self.encoder.fit_transform(self.X_train)
            self.X_test = self.encoder.transform(self.X_test)
            print("Data Preprocessing Completed")
        except Exception as e:
            print("Error in preprocessing ...",e)

    def train_model(self):
        """
        this function is used for training the model
        """
        try:
            print("Training model ...")
            self.model = RandomForestClassifier(n_estimators=100,random_state=42)
            self.model.fit(self.X_train, self.y_train)
            print("\nModel Training Completed")
        except Exception as e:
            print("Error in training the model ...",e)

    def evaluate_model(self):
        """
        this function is used for evaluating the model
        """
        try:
            print("Evaluating model ...")
            y_pred = self.model.predict(self.X_test)
            accuracy = accuracy_score(self.y_test, y_pred)
            print("\nModel Accuracy:", accuracy)
            print("\nClassification Report")
            print(classification_report(self.y_test, y_pred))
            cm = confusion_matrix(self.y_test, y_pred)
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
            plt.title("Confusion Matrix")
            plt.xlabel("Predicted")
            plt.ylabel("Actual")
            plt.show()
            print("Model Evaluation Completed and Accuracy:", accuracy)
        except Exception as e:
            print("Error in evaluating model ...",e)

    def save_predictions(self, engine, table_name):
        """
        this function is used for saving predictions to db
        """
        try:
            print(f"Saving all predictions to {table_name}...")
            self.encoder = ce.OrdinalEncoder(cols=['buying', 'maint', 'lug_boot', 'safety'])
            self.X = self.encoder.fit_transform(self.X)
            predictions = self.model.predict(self.X)
            self.df["predicted_class"] = predictions
            self.df.to_sql(table_name, engine, if_exists="replace", index=False)
            print("Predictions saved successfully.")
        except Exception as e:
            print("Error in saving predictions:", e)

def main():
    try:
        server = "localhost"
        database = "kamlesh"
        loader = MSSQLDataLoader(server, database)
        df = loader.load_table("car_evaluation")
        print("Original Data:")
        print(df.head())
        pipeline = CarEvaluationPipeline(df)
        pipeline.perform_eda()
        pipeline.preprocess(df)
        pipeline.train_model()
        pipeline.evaluate_model()
        pipeline.save_predictions(
            engine=loader.engine,
            table_name="car_evaluation",
        )
    except Exception as e:
        print("Error in executing model ...",e)


if __name__ == "__main__":
    main()
