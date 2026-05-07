import numpy as np
import pandas as pd
import math
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score,classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')


def load_data():
    """This function loads data from csv file
    :return: dataframe
    """
    data = pd. read_csv('bankloan.csv')
    print(data.head())
    return data

def data_stats(df):
    """This function compute statistics of data
    :return: None
    """
    print("Understanding the data")
    df.info()
    print(f'Shape: {df.shape}')
    print(f'Describe:\n{df.describe()}')
    numerical_col = df.select_dtypes(include=['int64', 'float64'])
    print(f'Numerical Columns :\n{numerical_col}')
    categorical_cols = df.select_dtypes(include=['str'])
    print(f'Categorical Columns :\n{categorical_cols}')

def cleaning_data(df):
    """This function cleans data handle missing values and duplicates
    :return: df
    """
    missing_percent = (df.isnull().sum() / len(df)) * 100
    print(f'Missing Percentage:\n{missing_percent}')
    print("Observation : No missing values found")
    duplicates = df.duplicated().sum()
    print(f'Duplicate :\n{duplicates}')
    if duplicates > 0:
        df = df.drop_duplicates(keep='first')
    df = df[df['Experience'] >= 0]
    return df


def outlier_detection(df):
    """This function handle outliers detection using percentile
    :return: df
    """
    numcol = df[['Mortgage','CCAvg','Income','Age','ZIP.Code']].columns.tolist()
    num_cols = len(numcol)
    rows = math.ceil(num_cols / 3)
    cols_per_row = 3
    plt.figure(figsize=(20, rows * 4))
    for i, col in enumerate(numcol):
        plt.subplot(rows, cols_per_row, i + 1)
        sns.boxplot(y=df[col], color='skyblue')
        plt.title(f'Boxplot of {col}')
    plt.tight_layout()
    plt.show()

    cols_to_fix = ['Income', 'CCAvg', 'Mortgage']
    df_new = df.copy()

    for col in cols_to_fix:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        df_new[col] = df_new[col].clip(lower_bound, upper_bound)

        plt.figure(figsize=(15, 4))
        plt.subplot(1, 3, 1)
        sns.histplot(df[col], kde=True, color='red', label='Original')
        plt.title(f'{col} Before')
        plt.legend()
        plt.subplot(1, 3, 2)
        sns.histplot(df_new[col], kde=True, color='skyblue', label='Cleaned')
        plt.title(f'{col} After')
        plt.legend()
        plt.subplot(1, 3, 3)
        sns.boxplot(x=df_new[col], color='lightgreen')
        plt.title(f'{col} Cleaned Boxplot')
        plt.tight_layout()
        plt.show()
    return df_new

def split_data(df):
    """This function splits data
    :return: x_train, x_test, y_train, y_test
    """
    x = df.drop(columns=['CreditCard','ID'])
    y = df['CreditCard']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
    print(f"Training set size: {x_train.shape[0]} rows")
    print(f"Testing set size: {x_test.shape[0]} rows")
    print("Train Test Split Completed Successfully")
    return x_train, x_test, y_train, y_test

def train_model(x1, x2, y1, y2):
    """
    This function trains data using DecisionTreeClassifier and predict on test data and give accuracy, confusionmatrix, and classification report
    :return: None
    """
    dc = DecisionTreeClassifier(max_depth=10, min_samples_split=10, max_leaf_nodes=8, random_state=42,class_weight='balanced')
    dc.fit(x1, y1)
    y_predict = dc.predict(x2)
    print(accuracy_score(y2, y_predict))
    accuracy = (accuracy_score(y2, y_predict)) * 100
    print(f'Accuracy: {accuracy}')
    print(f"Classification Report:\n {classification_report(y2, y_predict)}")
    print(f'Confusion Matrix: {confusion_matrix(y2, y_predict)}')


if __name__ == "__main__":
    data = load_data()
    data_stats(data)
    clean_data = cleaning_data(data)
    outlier_detection(clean_data)
    x_train, x_test, y_train, y_test = split_data(clean_data)
    train_model(x_train, x_test, y_train, y_test)



