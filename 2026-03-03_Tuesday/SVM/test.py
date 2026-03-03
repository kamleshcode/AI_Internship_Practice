import matplotlib.pyplot  as plt
import numpy as np
import seaborn as sns
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV
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
        self.test_size = 0.2

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
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X,self.y,random_state = self.random_state,test_size=self.test_size)
        except Exception as e:
            print("Error in train_test_split : ",e)

    def train_model(self):
        """
        This function is used for training the model using grid search CV
        :return: None
        """
        try:
            print("Training model")
            self.pipeline= Pipeline(steps=[('preprocessing',self.preprocessing_pipeline),('training',self.model)])
            param_grid = {
                "training__C": [0.1, 1, 10, 100],
                "training__gamma": [1, 0.1, 0.01, 0.001],
                "training__kernel": ["linear", "poly", "rbf"]
            }
            print("Starting Grid Search CV... this may take a while...\n")
            grid_search = GridSearchCV(self.pipeline, param_grid, refit=True, verbose=2, cv=5)
            grid_search.fit(self.X_train, self.y_train)
            self.pipeline = grid_search.best_estimator_
            self.model = grid_search.best_estimator_.named_steps["training"]
            print(end=seperator)
            print(f"Best Parameters:{grid_search.best_params_}\n")
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
            y_pred = self.pipeline.predict(self.X_test)
            print("\nClassification Report:")
            print(classification_report(self.y_test, y_pred))
            print("Confusion Matrix:")
            print(confusion_matrix(self.y_test, y_pred))
        except Exception as e:
            print("Error in evaluate_model : ",e)

    def plot_decision_boundary(self):
        try:
            fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
            genders = ['Male', 'Female']  # List to iterate through
            for i, gender in enumerate(genders):
                ax = axes[i]
                x = np.linspace(self.X['age'].min(), self.X['age'].max(), 100)
                y = np.linspace(self.X['estimated_salary'].min(), self.X['estimated_salary'].max(), 100)
                xx, yy = np.meshgrid(x, y)
                grid = pd.DataFrame({'gender': gender, 'age': xx.ravel(), 'estimated_salary': yy.ravel()})
                Z = self.pipeline.predict(grid).reshape(xx.shape)
                ax.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
                subset = self.data[self.data['gender'] == gender]
                ax.scatter(subset['age'], subset['estimated_salary'], c=subset['purchased'], cmap='coolwarm',
                           edgecolors='k')
                ax.set_title(gender)
            plt.show()
        except Exception as e:
            print(f"Error: {e}")


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
    obj.plot_decision_boundary()

if __name__ == "__main__":
    main()