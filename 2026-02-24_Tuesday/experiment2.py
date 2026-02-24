import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score,classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np
import warnings
warnings.filterwarnings('ignore')

seperator = f"\n{"---"*70}\n"

def load_dataset():
    """Load the dataset
    :return : df
    """
    try:
        df = pd.read_csv('healthcare_stroke_data.csv')
        print(df.head())
        print("Data Loaded Successfully",end=seperator)
        return df
    except FileNotFoundError as e:
        print('Dataset Not Found',e)

def understanding_data(df):
    """This function is used to understand the data
    :return : None
    """
    print("Understanding the data")
    print(f'Shape: {df.shape}')
    print('Information:',df.info())
    numerical_col = df.select_dtypes(include=['int64', 'float64'])
    print(f'Numerical Columns :\n{numerical_col}')
    categorical_cols = df.select_dtypes(include=['str'])
    print(f'Categorical Columns :\n{categorical_cols}',end=seperator)

def cleaning_data(df):
    """This function is used to clean the data
    1.handling missing values
    2.handling duplicate values
    3.set age column datatype to int
    :return : df
    """
    df["age"] = np.ceil(df["age"].astype(int))
    null_percentage = (df.isnull().sum() / len(df)) * 100
    print(f'Null Percentage: {null_percentage}')
    df.fillna(df['bmi'].median(), inplace=True)
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        df = df.drop_duplicates(keep='first')
    print(f'Duplicates: {duplicates}')
    print(f'Data Cleaned Successfully',end=seperator)
    return df

def checking_outliers(df):
    """This function is used to check the outliers using boxplot
    :return : None
    """
    num_col = df.select_dtypes(include=['int64', 'float64']).drop(columns=['id', 'stroke','hypertension','heart_disease']).columns
    plt.figure(figsize=(15, 5))
    for i, col in enumerate(num_col):
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        print(f"Feature: {col:18} | Total Outliers: {len(outliers)}")
        plt.subplot(2, 2, i + 1)
        sns.boxplot(y=df[col], color='skyblue')
        plt.title(f'Box Plot of {col}')
    plt.tight_layout()
    plt.show()


    num_col = ['bmi', 'avg_glucose_level']
    df_new = df.copy()

    for col in num_col:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        left_outliers = len(df[df[col] < lower_bound])
        right_outliers = len(df[df[col] > upper_bound])
        print(f"Feature: {col} ")
        print(f"Bounds: [{lower_bound:.2f}, {upper_bound:.2f}]")
        print(f"Outliers found: Left={left_outliers}, Right={right_outliers}")

        df_new[col] = df_new[col].clip(lower_bound, upper_bound)

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'Outlier Removal Analysis: {col}', fontsize=16)

        sns.histplot(df[col], kde=True, color='red', ax=axes[0, 0])
        axes[0, 0].set_title(f'{col} Original Distribution')

        sns.histplot(df_new[col], kde=True, color='skyblue', ax=axes[0, 1])
        axes[0, 1].set_title(f'{col} Cleaned Distribution')

        sns.boxplot(x=df[col], color='red', ax=axes[1, 0])
        axes[1, 0].set_title(f'{col} Original Boxplot')

        sns.boxplot(x=df_new[col], color='skyblue', ax=axes[1, 1])
        axes[1, 1].set_title(f'{col} Cleaned Boxplot')
        plt.tight_layout()
        plt.show()

    print("Checked Outliers and fixed it successfully",end=seperator)
    return df_new

def feature_engineering(df):
    """This function perform encoding on categorical columns
    :return : df
    """
    # Note : Decision Trees are scale-invariant; they split based on rank/order, so manual scaling (Standardization/Normalization) is not required.
    #Label Encoding
    le = LabelEncoder()
    binary_cols = ['ever_married', 'Residence_type', 'gender']
    for col in binary_cols:
        df[col] = le.fit_transform(df[col])

    #One hot encoding
    df = pd.get_dummies(df, columns=['work_type', 'smoking_status'], drop_first=True).astype(int)

    if 'id' in df.columns:
        df.drop(columns=['id'],inplace=True)
    print("Feature Engineering Completed Successfully",end=seperator)
    return df

def train_split_data(df):
    """This function is used to split the data in train and test
    :return : x_train, x_test, y_train, y_test
    """
    x = df.drop(columns=['stroke'])
    y = df['stroke']
    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3, random_state=42)
    print(f"Training set size: {x_train.shape[0]} rows")
    print(f"Testing set size: {x_test.shape[0]} rows")
    print("Train Test Split Completed Successfully",end=seperator)
    return x_train, x_test, y_train, y_test

def train_and_predict(x1, x2, y1, y2):
    """This function is used to train the model and predict the result
    :return : None
    """
    dc = DecisionTreeClassifier(criterion='gini', max_depth=15, min_samples_split=15, max_leaf_nodes=5, random_state=42,class_weight='balanced',ccp_alpha=0.001)
    dc.fit(x1, y1)
    y_predict = dc.predict(x2)
    print(accuracy_score(y2, y_predict))
    accuracy = (accuracy_score(y2, y_predict)) * 100
    print(f'Accuracy: {accuracy}')
    print(f"Classification Report:\n {classification_report(y2, y_predict)}")
    print(f'Confusion Matrix: {confusion_matrix(y2, y_predict)}')

def plot(df):
    """This function is used to plot correlation of numerical columns"""
    corr = df.corr()
    sns.heatmap(corr, annot=True)
    plt.show()

if __name__ == '__main__':
    data = load_dataset()
    understanding_data(data)
    clean_data = cleaning_data(data)
    checking_outliers(clean_data)
    final_data=feature_engineering(clean_data)
    x_train, x_test, y_train, y_test = train_split_data(final_data)
    train_and_predict(x_train, x_test, y_train, y_test)
    plot(final_data)



