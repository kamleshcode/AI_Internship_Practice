import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
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


class EsportsPlayerEvaluationPipeline:
    def __init__(self, df):
        self.df = df
        self.model = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.num_cols= None

    def perform_eda(self):
        """
        This function is used for EDA on df
        """
        try:
            print("Performing EDA ...")
            print(f"\nDataset Shape: {self.df.shape}")
            print('Information about dataset:\n')
            self.df.info()
            self.num_cols = self.df.select_dtypes(include=['int64', 'float64']).drop(columns=['record_id','player_id']).columns
            plt.figure(figsize=(15, 10))
            plt.suptitle('Distribution of Numerical Features', fontsize=16)
            for i, col in enumerate(self.num_cols):
                plt.subplot(4, 2, i + 1)
                sns.kdeplot(self.df[col], color="red", fill=True)
                plt.xlabel(col)
                plt.ylabel('Density')
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print("Error in performing eda ...", e)

    def preprocess(self):
        try:
            print("Preprocessing ...")
            cat_cols = self.df.select_dtypes(include=['str']).drop(columns=['team_name']).columns.to_list()
            encoder = OneHotEncoder(sparse_output=False, drop='first')
            encoded_cols = encoder.fit_transform(self.df[cat_cols])
            encoded_df = pd.DataFrame(encoded_cols,columns=encoder.get_feature_names_out(cat_cols),index=self.df.index)
            self.df = self.df.drop(columns=cat_cols)
            self.df = pd.concat([self.df, encoded_df],axis=1)
            self.X = self.df.drop(columns=['player_id', 'record_id', 'mvp_award', 'team_name'])
            self.y = self.df['mvp_award']
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42,stratify=self.y)
            print(f"Preprocessing Complete. Features: {self.X.shape[1]}")
        except Exception as e:
            print("Error in preprocessing ...", e)

    def train_model(self):
        """
        this function is used for training the model
        """
        try:
            print("Training model ...")
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
            self.model.fit(self.X_train, self.y_train)
            print("\nModel Training Completed")
        except Exception as e:
            print("Error in training the model ...", e)

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
            plt.figure(figsize=(15, 10))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
            plt.title("Confusion Matrix")
            plt.xlabel("Predicted")
            plt.ylabel("Actual")
            plt.show()
            print("Model Evaluation Completed and Accuracy:", accuracy)
        except Exception as e:
            print("Error in evaluating model ...", e)

    def save_predictions(self, engine, table_name):
        """
        this function is used for saving predictions to database
        """
        try:
            print(f"Saving all predictions to {table_name}...")
            self.df["predicted_mvp"] = self.model.predict(self.X)
            self.df.to_sql(table_name, engine, if_exists="replace", index=False)
            print("Predictions saved successfully.")
        except Exception as e:
            print("Error in saving predictions:", e)


def main():
    try:
        server = "localhost"
        database = "kamlesh"
        loader = MSSQLDataLoader(server, database)
        df = loader.load_table("esports_player_performance_results")
        print("Original Data:")
        print(df.head())
        pipeline = EsportsPlayerEvaluationPipeline(df)
        pipeline.perform_eda()
        pipeline.preprocess()
        pipeline.train_model()
        pipeline.evaluate_model()
        pipeline.save_predictions(loader.engine, "esports_player_performance_results")

    except Exception as e:
        print("Error in executing model ...",e)


if __name__ == "__main__":
    main()
