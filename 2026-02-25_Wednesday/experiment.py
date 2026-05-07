import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
import warnings
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder

warnings.filterwarnings('ignore')
seperator = f"\n{"---"*70}\n"

class DTRegressor:
    """
    This class perform end-to-end regression analysis using a Decision Tree.
    This class handles data loading, preprocessing, exploratory data
    analysis (EDA), pipeline construction with feature encoding, model
    training, and evaluation.
    """
    def __init__(self,file_path):
        """
        Initializes the DTRegressor with the path to the dataset.
        :param : file_path
        """
        self.file_path = file_path
        self.df = None
        self.X = None  # Features
        self.y = None  # Target
        self.Xtrain = None
        self.ytrain = None
        self.Xtest = None
        self.ytest = None
        self.pipeline = None
        self.random_state = 42
        self.cc_alpha = 0.01
        self.min_samples_split = 10
        self.max_depth = 4
        self.test_size = 0.3

    def load_dataset(self):
        """
        Reads the CSV file into a pandas DataFrame
        """
        try:
            print("Loading dataset...")
            self.df = pd.read_csv(self.file_path)
            print("Dataset loaded below are top 5 rows :\n")
            print(self.df.head(),end=seperator)
        except Exception as e:
            print(f'Error in loading dataset : {e}')

    def data_stats(self):
        """
        Prints basic statistical information about the dataset, including shape, info, and descriptive statistics.
        """
        try:
            print("Data stats :")
            print(f'Shape: {self.df.shape}')
            print(f'Information :{self.df.info()}')
            print(f'Description :{self.df.describe()}',end=seperator)
        except Exception as e:
            print(f'Error in checking Data Stats : {e}')

    def preprocess_data(self):
        """
        Cleans the dataset by identifying null values and dropping duplicates.
        """
        try:
            print("Preprocessing dataset...")
            print(f'Finding Null Values : {self.df.isnull().sum()}')
            print("Observation : No null values in dataset")
            print(f'Finding Duplicates : {self.df.duplicated().sum()}')
            self.df.drop_duplicates(keep='last', inplace=True)
            print("Observation : 1 duplicate was found that was dropped.\n",end=seperator)
        except Exception as e:
            print(f'Error in preprocessing dataset : {e}')

    def eda(self):
        """
         Performs Exploratory Data Analysis by generating Histograms, Boxplots, and Countplots for numerical and categorical features.
        """
        try:
            print("EDA :")
            #Numerical Column Analysis
            num_df = self.df.select_dtypes(include=['int64', 'float64'])
            print(f'Num Columns : {num_df.columns.tolist()}')
            target_col = self.df['charges']
            n_cols = 2
            n_rows = math.ceil(len(num_df.columns) / n_cols)
            plt.figure(figsize=(15, n_rows * 5))
            plt.suptitle("Numerical Column Distribution", fontsize=16)
            for i, col in enumerate(num_df.columns):
                plt.subplot(n_rows, n_cols, i + 1)
                sns.histplot(self.df[col], kde=True)
                plt.title(f'Distribution of {col}')
            plt.tight_layout()
            plt.show()

            plt.figure(figsize=(15, n_rows * 5))
            plt.suptitle("Box Plots ", fontsize=16)
            for i,col in enumerate(num_df.columns):
                plt.subplot(n_rows, n_cols, i + 1)
                sns.boxplot(self.df[col],palette="bright")
                plt.title(f'{col}')
            plt.tight_layout()
            plt.show()

            plt.figure(figsize=(15, n_rows * 5))
            corr=self.df.corr(numeric_only=True)
            sns.heatmap(corr, annot=True)
            plt.show()

            # Categorical Column Analysis
            cat_df = self.df.select_dtypes(include=['object'])
            print(f'Categorical Columns : {cat_df.columns.tolist()}')
            if not cat_df.empty:
                n_rows_cat = math.ceil(len(cat_df.columns) / n_cols)
                plt.figure(figsize=(15, n_rows_cat * 5))
                plt.suptitle("Categorical Column Distribution", fontsize=16)

                for i, col in enumerate(cat_df.columns):
                    plt.subplot(n_rows_cat, n_cols, i + 1)
                    sns.countplot(x=col, data=self.df)
                    plt.title(f'Count of {col}')

                plt.tight_layout()
                plt.show()
            else:
                print("No categorical columns found.")
            print("Exploratory Data Analysis Done....",end=seperator)
        except Exception as e:
            print("Error :",e)

    def train_test_split(self):
        """
        Splits the dataset into training and testing sets.Target is assumed to be the last column of the dataFrame.
        """
        try:
            print("Training and testing set...")
            self.X = self.df.iloc[:,:-1]
            self.y = self.df.iloc[:,-1]
            self.Xtrain, self.Xtest, self.ytrain, self.ytest = train_test_split(self.X,self.y,test_size=self.test_size,random_state=self.random_state)
            print("Train set created successfully.",end=seperator)
        except Exception as e:
            print(f'Error in train_test_split : {e}')

    def build_pipeline(self):
        """
        Constructs a scikit-learn pipeline consisting of a ColumnTransformer (for One-Hot Encoding) and a DecisionTreeRegressor.
        """
        try:
            print("Constructing pipeline ...")
            categorical_cols = self.df.select_dtypes(include=['object']).columns.tolist()
            transformer = ColumnTransformer(transformers=[('cat', OneHotEncoder(handle_unknown='ignore',drop="first"), categorical_cols)],remainder='passthrough')
            model = DecisionTreeRegressor( max_depth=self.max_depth, min_samples_split=self.min_samples_split, ccp_alpha=self.cc_alpha, random_state=self.random_state)
            self.pipeline = make_pipeline(transformer, model)
            print("Pipeline created successfully.",end=seperator)
        except Exception as e:
            print(f'Error in building pipeline: {e}')

    def train_model(self):
        """
        Fits the constructed pipeline onto the training data.
        """
        try:
            print("Fitting pipeline ...")
            self.pipeline.fit(self.Xtrain, self.ytrain)
            print("Model trained successfully.",end=seperator)
        except Exception as e:
            print(f'Error in training model: {e}')

    def evaluate_model(self):
        """
        Predicts target values for the test set and prints evaluation metrics including MAE and R2 Score.
        """
        try:
            print("Evaluating model ...")
            ypred = self.pipeline.predict(self.Xtest)
            mae = mean_absolute_error(self.ytest, ypred)
            mse = mean_squared_error(self.ytest, ypred)
            r2 = r2_score(self.ytest, ypred)
            print(f'Mean Absolute Error (MAE): {mae}')
            print(f'Mean Squared Error (MSE): {mse}')
            print(f'R2 Score: {r2}',end=seperator)
            print("Model evaluated successfully.",end=seperator)
        except Exception as e:
            print(f'Error in evaluating model: {e}')

    def tree_depth(self):
        """
        Calculate the actual and optimal depth of the tree.
        """
        trained_tree = self.pipeline.named_steps['decisiontreeregressor']
        actual_depth = trained_tree.get_depth() # 15
        print(f"The tree grew to a depth of: {actual_depth}")
        param_grid = {'decisiontreeregressor__max_depth': range(1, 20)}
        grid_search = GridSearchCV(self.pipeline, param_grid, cv=5, scoring='r2') # 4
        grid_search.fit(self.Xtrain, self.ytrain)
        print(f"Calculated Optimal Depth: {grid_search.best_params_['decisiontreeregressor__max_depth']}")

def main():
    """
    Main execution block to instantiate the class and run the workflow
    """
    filepath = "../2026-02-19_Thursday/insurance.csv"
    obj = DTRegressor(filepath)
    obj.load_dataset()
    obj.preprocess_data()
    obj.eda()
    obj.train_test_split()
    obj.build_pipeline()
    obj.train_model()
    obj.evaluate_model()
    obj.tree_depth()


if __name__ == '__main__':
    main()