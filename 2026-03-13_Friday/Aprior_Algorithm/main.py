import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mlxtend.frequent_patterns import apriori, association_rules

class AprioriAlgorithm:
    """
    Initializes the class with the dataset path and empty attributes.
    """
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = None
        self.basket = None
        self.rules = None
        self.top_items = None
        self.support=0.001
        self.confidence=0.05

    def load_data(self):
        """
        Loads the CSV file into a Pandas DataFrame and parses the 'Date' column.
        """
        try:
            # day-first=True to handle dd-mm-yyyy format correctly
            self.df = pd.read_csv(self.file_path, parse_dates=['Date'], dayfirst=True)
            print(f"Data Loaded. Shape: {self.df.shape}")
        except FileNotFoundError:
            print("Error: File not found.")

    def plot_distribution(self):
        """
        Displays a Seaborn Barplot of the top 10 most frequent products.
        """
        plt.figure(figsize=(12, 6))
        sns.barplot(x=self.top_items.values, y=self.top_items.index, hue=self.top_items.index, palette="viridis",
                    legend=False)
        plt.title("Top 10 Selling Products", fontsize=15)
        plt.xlabel("Count")
        plt.ylabel("Products")
        plt.show()

    def get_stats(self):
        """
        Prints basic data overview
        """
        print("\n--- Data Statistics ---")
        print(f'Shape of Data: {self.df.shape}')
        print(f'Null values in Data: {self.df.isnull().sum()}')
        print(f'Describe Data: {self.df.describe()}')
        unique_items = self.df['itemDescription'].nunique()
        print(f"Total Unique Products: {unique_items}")
        self.top_items = self.df['itemDescription'].value_counts().head(10)
        self.plot_distribution()

    def preprocess_data(self):
        """
        Pivots raw transaction logs into a binary (1/0) Market Basket matrix.
        Groups data by Member/Date, converts items to columns, and maps counts to 1 if present or 0 if absent.
        """
        print("\nPreprocessing into Market Basket format...")
        self.basket = (self.df.groupby(['Member_number', 'Date', 'itemDescription'])['itemDescription']
                       .count().unstack().fillna(0))

        self.basket = self.basket.map(lambda x: 1 if x > 0 else 0)
        print(f"Basket created. Transactions: {len(self.basket)}")

    def extract_associations(self):
        """
        Finds frequent itemsets via Apriori and filters for Lift > 1
        """
        print(f"\nRunning Apriori (Support={self.support})...")
        frequent_itemsets = apriori(self.basket.astype(bool), min_support=self.support, use_colnames=True)
        print(f"Frequent Itemsets:\n{frequent_itemsets}")

        rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=self.confidence)
        print(f"Rules:\n{rules}")
        # If Lift > 1: The items are positively correlated (Buying A actually increases the chance of B).
        # If Lift = 1: The items are independent (Buying A and B together is just a coincidence).
        # If Lift < 1: The items are negatively correlated (Buying A makes you LESS likely to buy B).
        self.rules = rules[rules['lift'] > 1].sort_values('lift', ascending=False)
        print(f"Rules Found: {len(self.rules)}")

    def plot_visualizations(self):
        """
        plot showing Support vs. Confidence
        """
        if self.rules is None or self.rules.empty:
            print("No rules to visualize.")
            return

        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=self.rules, x="support", y="confidence",
                        hue="lift", size="lift", palette="rocket", sizes=(50, 200))
        plt.title("Rules: Support vs Confidence")
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.show()

    def display_rules(self):
        """
        Prints the top 10 discovered "if-then" association rules.
        """
        if self.rules is not None and not self.rules.empty:
            print(f"\n--- Top 10 Association Rules ---")
            print(self.rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head(10))

def main():
    """
    Standard driver function to execute the Grocery Association Mining pipeline.
    """
    try:
        file_path = 'Groceries_dataset.csv'
        obj = AprioriAlgorithm(file_path)
        obj.load_data()
        obj.get_stats()
        obj.preprocess_data()
        obj.extract_associations()
        obj.display_rules()
        obj.plot_visualizations()
    except Exception as e:
        print(f'Error in main function: {e}')

if __name__ == "__main__":
    main()