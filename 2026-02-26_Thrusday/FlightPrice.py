import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import plot_tree
import warnings
warnings.filterwarnings('ignore')

seperator = f"\n{"---"*70}\n"

logging.basicConfig(
    filename='./flight_price.log',
    filemode='w',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

class RandomForest:
    def __init__(self,file_path):
        self.file_path = file_path
        self.df = pd.DataFrame()
        self.df_new=pd.DataFrame()
        self.num_col=None
        self.cat_col=None
        self.X=None
        self.y=None
        self.Xtrain, self.Xtest, self.ytrain, self.ytest = None, None, None, None
        self.test_size=0.3
        self.random_state=42
        self.n_estimators=50
        self.max_depth=10
        self.min_samples_leaf=5
        self.verbose=2
        self.pipeline=None

    def load(self):
        """
        Loads data from csv file
        """
        try:
            logger.info("Loading data...")
            self.df = pd.read_csv(self.file_path)
            print(self.df.head())
            logger.info("Data loaded successfully.")
        except Exception as e:
            logger.error('Error in loading data : ',e)

    def data_stats(self):
        """
        Prints basic statistical information about the dataset, including shape, info, and descriptive statistics.
        """
        try:
            logger.info("Analyzing data")
            print("Data stats :")
            print(f'Shape: {self.df.shape}')
            print(f'Information :\n')
            self.df.info()
            print(f'Description :\n{self.df.describe()}', end=seperator)
            logger.info("Data analyzed done..")
        except Exception as e:
            logger.error(f'Error in checking Data Stats : {e}')

    def preprocessing(self):
        """
        Cleans the dataset by identifying null values and dropping duplicates.
        """
        try:
            logger.info("Preprocessing data...")
            print(f'Finding Null Values : {self.df.isnull().sum().sum()}')
            print(f'Finding Duplicates : {self.df.duplicated().sum()}')
            self.df.drop(self.df.columns[0], axis=1, inplace=True)
            print(f'Drop first columns')
            print(f'Avg price:{self.df['price'].mean()}',end=seperator)
            logger.info("Preprocessing done..")
        except Exception as e:
            logger.error(f'Error in preprocessing dataset : {e}')

    def eda(self):
        try:
            logger.info("EDA analysis started....")
            self.num_col=self.df.select_dtypes(include=['int64','float64'])
            print(f'Numerical Columns: {self.num_col.columns}')
            self.cat_col=self.df.select_dtypes(include=['object']).drop(columns=['flight'])
            print(f'Categorical Columns: {self.cat_col.columns}')

            plt.figure(figsize=(10, 10))
            for i, col in enumerate(self.cat_col.columns):
                plt.subplot(4, 2, i + 1)
                sns.countplot(x=col, data=self.df)
                plt.title(f'Count of {col}')
            plt.tight_layout()
            plt.show()

            plt.figure(figsize=(10, 10))
            plt.suptitle("Numerical Column Distribution", fontsize=16)
            for i, col in enumerate(self.num_col.columns):
                plt.subplot(2, 2, i + 1)
                sns.histplot(self.df[col], kde=True)
            plt.tight_layout()
            plt.show()

            #Boxplot
            plt.figure(figsize=(8, 10))
            plt.suptitle("Box Plots ", fontsize=16)
            for i, col in enumerate(self.num_col.columns):
                plt.subplot(3, 5, i + 1)
                sns.boxplot(self.df[col], orient='h')
                plt.title(f'{col}')
            plt.tight_layout()
            plt.show()
            logger.info("Exploratory Data Analysis Done Successfully...")
        except Exception as e:
            logger.error(f'Error in exploratory Data Analysis : {e}')

    def handling_outlier(self):
        try:
            logger.info("Outlier Removal Analysis started....")
            self.num_col=['duration','days_left']
            self.df_new=self.df.copy()
            for col in self.num_col:
                q1 = self.df[col].quantile(0.25)
                q3 = self.df[col].quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                left_outliers = len(self.df[self.df[col] < lower_bound])
                right_outliers = len(self.df[self.df[col] > upper_bound])
                print(f"Feature: {col} ")
                print(f"Bounds: [{lower_bound:.2f}, {upper_bound:.2f}]")
                print(f"Outliers found: Left={left_outliers}, Right={right_outliers}")

                self.df_new[col] = np.where(self.df_new[col] < lower_bound, lower_bound, np.where(self.df_new[col] > upper_bound, upper_bound, lower_bound))

                fig, axes = plt.subplots(2, 2, figsize=(14, 10))
                fig.suptitle(f'Outlier Removal Analysis: {col}', fontsize=16)

                sns.histplot(self.df[col], kde=True, color='red', ax=axes[0, 0])
                axes[0, 0].set_title(f'{col} Original Distribution')

                sns.histplot(self.df_new[col], kde=True, color='skyblue', ax=axes[0, 1])
                axes[0, 1].set_title(f'{col} Cleaned Distribution')

                sns.boxplot(x=self.df[col], color='red', ax=axes[1, 0])
                axes[1, 0].set_title(f'{col} Original Boxplot')

                sns.boxplot(x=self.df_new[col], color='skyblue', ax=axes[1, 1])
                axes[1, 1].set_title(f'{col} Cleaned Boxplot')
                plt.tight_layout()
                plt.show()
                logger.info("Outlier Removal Analysis Done Successfully...")
        except Exception as e:
            logger.error(f'Error in handling_outlier : {e}')

    def train_test_split(self):
        try:
            logger.info("Started Splitting data......")
            cols_to_drop = ['price','flight']
            self.X = self.df.drop(columns=cols_to_drop)
            self.y = self.df.iloc[:,-1]
            self.Xtrain, self.Xtest, self.ytrain, self.ytest = train_test_split(self.X, self.y, test_size=self.test_size, random_state=self.random_state)
            logger.info("Train set created successfully.")
        except Exception as e:
            logger.error(f'Error in train_test_split : {e}')

    def build_pipeline(self):
        try:
            logger.info("Constructing pipeline ...")
            categorical_cols = self.X.select_dtypes(include=['object']).columns.tolist()
            transformer = ColumnTransformer(
                transformers=[('cat', OneHotEncoder(handle_unknown='ignore', drop="first"), categorical_cols)],
                remainder='passthrough'
            )
            model = RandomForestRegressor(n_estimators=self.n_estimators, max_depth=self.max_depth,
                                           random_state=self.random_state, min_samples_split=self.min_samples_leaf,verbose=self.verbose)
            self.pipeline = Pipeline(steps=[('preprocessor', transformer),('model', model)])
            logger.info("Pipeline constructed successfully.")
        except Exception as e:
            logger.error(f'Error in build_pipeline : {e}')

    def train_model(self):
        try:
            logger.info("Training model...")
            self.pipeline.fit(self.Xtrain, self.ytrain)
            logger.info("Model trained successfully.")
        except Exception as e:
            logger.error(f'Error in train_model : {e}')

    def model_evaluation(self):
        try:
            logger.info("Evaluating model ...")
            ypred = self.pipeline.predict(self.Xtest)
            mae = mean_absolute_error(self.ytest, ypred)
            mse = mean_squared_error(self.ytest, ypred)
            r2 = r2_score(self.ytest, ypred)
            print(f'Mean Absolute Error (MAE): {mae}')
            print(f'Mean Squared Error (MSE): {mse}')
            print(f'R2 Score: {r2}')
            logger.info("Model evaluated successfully.")
        except Exception as e:
            logger.error(f'Error in evaluating model: {e}')

def main():
    """
    Main execution block to instantiate the class and run the workflow
    """
    try:
        file_path = 'flight_price_prediction.csv'
        obj = RandomForest(file_path)
        obj.load()
        obj.data_stats()
        obj.preprocessing()
        # obj.eda()
        obj.handling_outlier()
        obj.train_test_split()
        obj.build_pipeline()
        obj.train_model()
        obj.model_evaluation()
    except Exception as e:
        print("Error in main : ",e)

if __name__ == "__main__":
    main()