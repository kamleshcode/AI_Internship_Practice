import matplotlib.pyplot  as plt
import seaborn as sns
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.svm import SVC
import warnings
warnings.filterwarnings("ignore")

seperator = f'\n\n{'--'*50}\n'

class SupportVectorClassifier:
    """
    This class is used for the Support Vector Machine model
    """
    def __init__(self, filepath):
        self.filepath = filepath
        self.data = None
        self.num_col = None
        self.cat_col = None
        self.encoder = OneHotEncoder()
        self.scaler = StandardScaler()
        self.model = SVC()
        self.preprocessing_pipeline = None
        self.pipeline = None
        self.X = None
        self.y = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.random_state = 42

    def load_data(self):
        """
        This function is used for loading the data
        :return: None
        """
        try:
            self.data = pd.read_csv(self.filepath)
            print("Data loaded")
            print(self.data.head(),end=seperator)
        except Exception as e:
            print("Error loading data : ",e)

    def preprocessing_data(self):
        """
        This function is used for preprocessing the data checking null and duplicate values
        :return: None
        """
        try:
            print("Data Stats")
            self.data.info()
            print(self.data.describe())
            print(f'Checking Null values:\n {self.data.isnull().sum()}')
            print(f'Checking Duplicate values:\n {self.data.duplicated().sum()}')
            print("Extracting independent and dependent variables")
            self.num_col = self.data.select_dtypes(include=['int64','float64']).drop(columns=['user_id','purchased']).columns
            print(f'Numerical Columns : {self.num_col}')
            self.cat_col = self.data.select_dtypes(include=['str']).columns
            print(f'Categorical Columns : {self.cat_col}',end=seperator)
        except Exception as e:
            print("Error checking data : ",e)

    def eda(self):
        """
        This function is used for check distribution of data
        :return: None
        """
        try:
            print("performing EDA")
            df = self.data.copy().drop(columns=['user_id'])
            corr = df.corr(numeric_only=True)
            plt.title('Correlation Matrix', fontsize=20, fontweight='bold')
            sns.heatmap(corr, annot=True)
            plt.show()

            sns.pairplot(df, hue="purchased")
            plt.show()
            print("EDA completed ...")
        except Exception as e:
            print("Error in analytics : ",e)

    def checking_outliers(self):
        """
        This function is used for checking outliers in our features
        :return: None
        """
        try:
            for col in self.num_col:
                q1 = self.data[col].quantile(0.25)
                q3 = self.data[col].quantile(0.75)
                iqr = q3 - q1
                lb = q1 - 1.5 * iqr
                ub = q3 + 1.5 * iqr
                outliers = self.data[(self.data[col] < lb) | (self.data[col] > ub)]
                print(f'Outliers in {col} : {len(outliers)}')
        except Exception as e:
            print("Error checking outliers : ",e)

    def build_pipeline(self):
        """
        This function is used for building the pipeline
        :return: None
        """
        self.preprocessing_pipeline = ColumnTransformer(transformers=[("categorical",self.encoder,["gender"]),("numeric",self.scaler,self.num_col)])
        print("Preprocessing pipeline created Successfully.")

    def train_test_split(self):
        """
        This function is used for splitting the data in dependent and independent variables and further into training and testing sets
        :return: None
        """
        try:
            self.X = self.data.iloc[:,:-1]
            self.y = self.data.iloc[:,-1]
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X,self.y,random_state = self.random_state)
        except Exception as e:
            print("Error in train_test_split : ",e)

    def train_model(self):
        """
        This function is used for training the model using pipeline
        :return: None
        """
        try:
            print("Training model")
            self.pipeline= Pipeline(steps=[('preprocessing',self.preprocessing_pipeline),('training',self.model)])
            self.pipeline.fit(self.X_train,self.y_train)
            print("Model Trained Completed ...")
        except Exception as e:
            print("Error in train_model : ",e)

    def evaluate_model(self):
        """
        This function is used for evaluating the model check accuracy and
        :return: None
        """
        try:
            print("Evaluating model")
            accuracy = self.pipeline.score(self.X_test,self.y_test)
            print(f'Accuracy : {accuracy*100}%')
            confusion_matrix = self.pipeline.confusion_matrix(self.X_test,self.y_test)
            sns.pairplot(confusion_matrix)
            plt.show()
        except Exception as e:
            print("Error in evaluate_model : ",e)

def main():
    file_path = "../data/user_data.csv"
    obj = SupportVectorClassifier(file_path)
    obj.load_data()
    obj.preprocessing_data()
    obj.eda()
    obj.checking_outliers()
    obj.build_pipeline()
    obj.train_test_split()
    obj.train_model()
    obj.evaluate_model()

if __name__ == "__main__":
    main()