import joblib
import matplotlib.pyplot  as plt
import seaborn as sns
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, auc
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,OneHotEncoder
import warnings
warnings.filterwarnings("ignore")

seperator = f'\n\n{'--'*50}\n'

class KNNClassifier:
    """
    This class is used for the Support Vector Machine model
    """
    def __init__(self, filepath):
        self.filepath = filepath
        self.data = None
        self.num_col = None
        self.cat_col = None
        self.model = None
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
            self.data.drop(columns=['user_id'], inplace=True)
        except Exception as e:
            print("Error checking data : ",e)

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

    def build_pipeline(self):
        """
        This function is used for building the pipeline
        :return: None
        """
        self.num_col = self.data.select_dtypes(include=['int64', 'float64']).drop(columns=['purchased']).columns.tolist()
        self.cat_col = self.data.select_dtypes(include=['str']).columns.tolist()
        num_pipeline = StandardScaler()
        cat_pipeline = OneHotEncoder()
        preprocessor = ColumnTransformer(transformers=
        [
            ("categorical",cat_pipeline,self.cat_col),
            ("numeric",num_pipeline,self.num_col)
        ],
            remainder='passthrough')

        self.model = KNeighborsClassifier()
        self.pipeline = Pipeline(steps = [
            ("preprocessor", preprocessor),
            ("classifier", self.model)
        ])
        print("Preprocessing pipeline created Successfully.")

    def train_model(self):
        """
        This function is used for training the model using grid search CV
        :return: None
        """
        try:
            params_grid = {
                "classifier__n_neighbors": [3, 5, 7, 9, 11, 15],
                "classifier__weights": ["uniform", "distance"],
                "classifier__metric" : ["euclidean", "manhattan", "minkowski"]
            }
            grid_search = GridSearchCV(self.pipeline, param_grid=params_grid,cv=5,verbose=2)
            grid_search.fit(self.X_train, self.y_train)
            self.pipeline = grid_search.best_estimator_
            self.model = grid_search.best_estimator_.named_steps["classifier"]
            print(f'Best Parameters : {grid_search.best_params_}')
        except Exception as e:
            print("Error in train_model : ",e)

    def evaluate_model(self):
        """
        This function is used for evaluating the model check accuracy and
        :return: None
        """
        try:
            print("Evaluating model")
            y_pred = self.pipeline.predict(self.X_test)
            accuracy = accuracy_score(self.y_test,y_pred)
            print(f'Accuracy : {accuracy*100}%')
            print("\nClassification Report:")
            print(classification_report(self.y_test, y_pred))
            print("Confusion Matrix:")
            print(confusion_matrix(self.y_test, y_pred))
            y_score = self.pipeline.predict_proba(self.X_test)[:, 1]
            fpr, tpr, _ = roc_curve(self.y_test, y_score)
            roc_auc = auc(fpr, tpr)
            plt.figure(figsize=(6, 4))
            plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curve')
            plt.legend()
            plt.show()
        except Exception as e:
            print("Error in evaluate_model : ",e)

    def save_model(self):
        try:
            joblib.dump(self.pipeline, "../2026-03-05_Thursday/model.pkl")
            print("Model Saved Successfully.")
        except Exception as e:
            print("Error in save_model : ",e)

def main():
    file_path = "../2026-03-03_Tuesday/data/user_data.csv"
    obj = KNNClassifier(file_path)
    obj.load_data()
    obj.preprocessing_data()
    obj.eda()
    obj.checking_outliers()
    obj.train_test_split()
    obj.build_pipeline()
    obj.train_model()
    obj.evaluate_model()
    obj.save_model()

if __name__ == "__main__":
    main()