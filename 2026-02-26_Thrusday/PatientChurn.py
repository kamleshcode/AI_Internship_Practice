import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report,confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder
import yaml
from imblearn.over_sampling import SMOTE

warnings.filterwarnings('ignore')

seperator = f"\n{"---"*70}\n"

class RandomForest:
    def __init__(self,config):
        self.config = config # store entire config dict
        self.file_path = self.config["data"]["file_path"]
        self.test_size = self.config["data"]["test_size"]
        self.df = pd.DataFrame()
        self.X=None
        self.y=None
        self.Xtrain = None
        self.ytrain = None
        self.Xtest = None
        self.ytest = None
        self.random_state = self.config["RandomForestClassifier"]["random_state"]
        self.n_estimators = self.config["RandomForestClassifier"]["n_estimators"]
        self.max_depth = self.config["RandomForestClassifier"]["max_depth"]
        self.min_samples_split = self.config["RandomForestClassifier"]["min_samples_split"]
        self.pipeline = None

    def load(self):
        """
        Loads data from csv file
        """
        try:
            self.df = pd.read_csv(self.file_path)
            print(self.df.head())
        except Exception as e:
            print('Error in loading data : ',e)

    def data_stats(self):
        """
        Prints basic statistical information about the dataset, including shape, info, and descriptive statistics.
        """
        try:
            print("Data stats :")
            print(f'Shape: {self.df.shape}')
            print(f'Information :{self.df.info()}')
            print(f'Description :{self.df.describe()}', end=seperator)
        except Exception as e:
            print(f'Error in checking Data Stats : {e}')

    def preprocessing(self):
        """
        Cleans the dataset by identifying null values and dropping duplicates.
        """
        try:
            print("Preprocessing dataset...")
            print(f'Finding Null Values : {self.df.isnull().sum().sum()}')
            print(f'Finding Duplicates : {self.df.duplicated().sum()}',end=seperator)
        except Exception as e:
            print(f'Error in preprocessing dataset : {e}')

    def eda(self):
        """
        Performs Exploratory Data Analysis by generating Histograms, Boxplots, and Countplots for numerical and categorical features.
        """
        try:
            print("EDA :")
            # Numerical Column Analysis
            num_df = self.df.select_dtypes(include=['int64', 'float64'])
            print(f'Num Columns : {num_df.columns.tolist()}')
            plt.figure(figsize=(10, 10))
            plt.suptitle("Numerical Column Distribution", fontsize=16)
            for i, col in enumerate(num_df.columns):
                plt.subplot(3, 5, i + 1)
                sns.histplot(self.df[col], kde=True)
            plt.tight_layout()
            plt.show()

            plt.figure(figsize=(8, 10))
            plt.suptitle("Box Plots ", fontsize=16)
            for i, col in enumerate(num_df.columns):
                plt.subplot(3, 5, i + 1)
                sns.boxplot(self.df[col],orient='h')
                plt.title(f'{col}')
            plt.tight_layout()
            plt.show()

            # Categorical Column Analysis
            cat_df = self.df.select_dtypes(include=['object']).drop(columns=['PatientID','Last_Interaction_Date'])
            print(f'Categorical Columns : {cat_df.columns.tolist()}')
            plt.figure(figsize=(15, 15))
            plt.suptitle("Categorical Column Distribution", fontsize=16)
            for i, col in enumerate(cat_df.columns):
                plt.subplot(2, 2, i + 1)
                sns.countplot(x=col, data=self.df)
                plt.title(f'Count of {col}')
            plt.tight_layout()
            plt.show()

            #Heatmap
            corr = self.df.corr(numeric_only=True)
            plt.figure(figsize=(15, 10))
            sns.heatmap(corr, annot=True, fmt=".2f")
            plt.title("Correlation Heatmap for Patient Churn")
            plt.show()
        except Exception as e:
            print("Error in EDA :",e)

    def train_test_split(self):
        """
        This function splits the dataset into train and test set.
        """
        try:
            cols_to_drop = ['PatientID', 'Last_Interaction_Date', 'Churned']
            self.X = self.df.drop(columns=cols_to_drop)
            self.y = self.df.iloc[:,-1]
            self.Xtrain, self.Xtest, self.ytrain, self.ytest = train_test_split(self.X, self.y, test_size=self.test_size, random_state=self.random_state)
            smote = SMOTE()
            self.Xtrain, self.Xtest = smote.fit_resample(self.Xtrain, self.Xtest)
            print("Train set created successfully.",end=seperator)
        except Exception as e:
            print(f'Error in train_test_split : {e}')

    def build_pipeline(self):
        """
        This function is used to build pipeline
        """
        print("Constructing pipeline ...")
        categorical_cols = self.df.select_dtypes(include=['object']).drop(columns=['PatientID','Last_Interaction_Date']).columns.tolist()

        transformer = ColumnTransformer(
            transformers=[('cat', OneHotEncoder(handle_unknown='ignore', drop="first"), categorical_cols)],
            remainder='passthrough'
        )
        model = RandomForestClassifier(n_estimators=self.n_estimators, max_depth=self.max_depth, random_state=self.random_state, min_samples_split=self.min_samples_split)
        self.pipeline = make_pipeline(transformer, model)
        print("Pipeline created successfully.", end=seperator)

    def train_model(self):
        """
        This function is used to fit model
        """
        try:
            print("Training model...")
            self.pipeline.fit(self.Xtrain, self.ytrain)
        except Exception as e:
            print(f'Error in train_model : {e}')

    def model_evaluation(self):
        """
        This function is used to evaluate model performance
        """
        try:
            print("Evaluating CLASSIFICATION Model ...")
            ypred = self.pipeline.predict(self.Xtest)
            acc = accuracy_score(self.ytest, ypred)
            print(f'Overall Accuracy: {acc:.2%}')
            print("\nClassification Report:")
            print(classification_report(self.ytest, ypred))
            print("\nConfusion Matrix:")
            print(confusion_matrix(self.ytest, ypred))
        except Exception as e:
            print(f'Error in model_evaluation : {e}')

def main():
    """
    Main execution block to instantiate the class and run the workflow
    """
    try:
        with open("params.yaml", "r") as f:
            config = yaml.safe_load(f)
        obj = RandomForest(config)
        obj.load()
        obj.data_stats()
        obj.preprocessing()
        obj.eda()
        obj.train_test_split()
        obj.build_pipeline()
        obj.train_model()
        obj.model_evaluation()
    except Exception as e:
        print("Error in main : ",e)

if __name__ == "__main__":
    main()