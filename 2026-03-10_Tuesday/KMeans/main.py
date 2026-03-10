import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler, LabelEncoder

class CustomerSegmentation:
    def __init__(self, file_path):
        self.X_scaled = None
        self.file_path = file_path
        self.data = None
        self.X = None
        self.model= None
        self.labels = None
        self.scaler = StandardScaler()

    def load_data(self):
        """
        This function is used for loading dataset
        :return: None
        """
        try:
            self.data = pd.read_csv(self.file_path)
        except FileNotFoundError:
            print('File not found')

    def data_analysis(self):
        """
        This function is for Exploratory Data analysis
        :return: None
        """
        try:
            print("Analyzing Data")
            plt.figure(figsize=(10,10))
            corr = self.data.corr(numeric_only=True)
            sns.heatmap(corr, annot=True, cmap="YlOrRd")
            plt.show()

            num_col =self.data.select_dtypes(include=['int64','float64']).columns
            print(num_col)
            for i, col in enumerate(num_col):
                plt.subplot(2,2,i+1)
                sns.boxplot(self.data[col])
            plt.show()

            plt.figure(figsize=(10,10))
            plt.scatter(self.data['Annual Income (k$)'],self.data['Spending Score (1-100)'])
            plt.xlabel("Annual Income (k$)")
            plt.ylabel("Spending Score (1-100)")
            plt.show()
        except Exception as e:
            print(e)

    def data_preprocessing(self):
        """
        This function is used to preprocess the data in which we have also scaled and encode the data
        :return: None
        """
        try:
            print("Preprocessing data...")
            print(f'Shape : {self.data.shape}')
            print('Information')
            self.data.info()
            print(f'Null Values : {self.data.isnull().sum()}')
            print(f'Duplicate Values : {self.data.duplicated().sum()}')

            self.data.drop(columns=['CustomerID'], inplace=True)

            le = LabelEncoder()
            self.data['Gender'] = le.fit_transform(self.data['Gender'])

            # Feature Selection
            self.X = self.data.copy()
            print(f"printljhk:{self.X.head()}")

            self.X_scaled = self.scaler.fit_transform(self.X) # it returns array
            print("features scaled successfully")
        except Exception as e:
                print(e)

    def find_optimal_k(self):
        """
        This function is used for finding the optimal k
        :return: None
        """
        try:
            wcss=[]
            for i in range(1,20):
                kmeans = KMeans(n_clusters=i, init='k-means++',random_state=42)
                kmeans.fit(self.X_scaled)
                wcss.append(kmeans.inertia_)

            plt.plot(range(1, 20), wcss, marker='o', color='red')
            plt.title('Elbow Method')
            plt.xlabel('Number of Clusters')
            plt.ylabel('WCSS')
            plt.show()
        except Exception as e:
            print("Error in finding optimal k : ",e)

    def train_and_plot(self, k=16):
        """
        this function is used for training the model and accept one parameter no. of clusters i.e k
        :param k: int
        :return: None
        """
        try:
            self.model = KMeans(n_clusters=k, init='k-means++', random_state=42)
            self.data['Cluster_formed'] = self.model.fit_predict(self.X_scaled)
            print(f'Number of clusters : {self.model.n_clusters}')
            print(f'Cost : {self.model.inertia_}')
            print("Cluster Distribution:")
            print(self.data['Cluster_formed'].value_counts())
            # Silhouette Score ranges from -1 to +1 (alternative to the Elbow Method to determine the most appropriate value for k )
            score = silhouette_score(self.X_scaled,self.data['Cluster_formed']) # Closer to 1 is better
            print(f'Silhouette Score: {score:.3f}')

        except Exception as e:
            print("Error in training/plotting:", e)

def main():
    """
    driver function to run the main logic
    :return:
    """
    try:
        file_path = 'Mall_Customers.csv'
        obj = CustomerSegmentation(file_path)
        obj.load_data()
        obj.data_analysis()
        obj.data_preprocessing()
        obj.find_optimal_k()

        obj.train_and_plot()
    except Exception as e:
        print("Error in executing main function", e)

if __name__ == '__main__':
    main()