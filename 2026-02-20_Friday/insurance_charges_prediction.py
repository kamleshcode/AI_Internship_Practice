import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import seaborn as sns
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("../2026-02-19_Thursday/insurance.csv")
df.head()

print(f'Shape of  dataset: {df.shape[0]} rows and {df.shape[1]} columns')
print(f'Columns of dataset: {list(df.columns)}')
missing = df.isnull().sum()
print(f'Missing values in dataset: {missing.sum()}')
duplicates = df.duplicated().sum()
print(f'Duplicate values in dataset: {duplicates.sum()}')
print(df.describe(include='str'))

df.drop_duplicates(keep="first",inplace=True)
df.count()

# One-Hot Encoding
df['sex'] = df['sex'].map({'female': 0, 'male': 1})
df['smoker'] = df['smoker'].map({'no': 0, 'yes': 1})
df = pd.get_dummies(df, columns=['region'], drop_first=True).astype(int)
df.head()

# Checking Outliers
# plt.figure(figsize=(8, 5))
# sns.boxplot(x=df['bmi'], color='green')
# plt.title('BMI Distribution and Outliers')
# plt.show()

#drop outliers
df = df[df['bmi'] < 47]

# log-transformed charges column
df['charges'] = np.log1p(df['charges'])

# plt.figure(figsize=(12, 5))
# plt.subplot(1, 2, 2)
# sns.boxplot(x=df['charges'], color='salmon')
# plt.title('Log Charges (Balanced)')
# plt.tight_layout()
# plt.show()

# Train-Test Split
X = df.drop('charges', axis=1)
y = df['charges']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print(f'X_train_scaled shape: {X_train_scaled.shape}')
print(f'X_test_scaled shape: {X_test_scaled.shape}')


# Model Training
model = LinearRegression()
model.fit(X_train_scaled, y_train)
print(f'Coefficient:{model.coef_}')
print(f'Intercept:{model.intercept_}')

# Evaluations
y_pred = model.predict(X_test_scaled)
residuals = y_test - y_pred
results = pd.DataFrame({
    "Actual" : y_test,
    "Predicted" : y_pred,
    "Residual" : residuals
})
print(results.head())

MAE = mean_absolute_error(y_test, y_pred)
R2 = r2_score(y_test, y_pred)
RMSE = np.sqrt(mean_squared_error(y_test, y_pred))
print(f'MAE : {MAE}\nR2: {R2 : .4f}\nRMSE: {RMSE : .2f}')

# Plots
plt.figure()
sns.histplot(y, kde=True)
plt.title("Distribution of Insurance Charges")
plt.xlabel("Insurance Charges")
plt.ylabel("Frequency")
plt.show()

plt.figure()
sns.pairplot(df, hue='smoker', palette='muted')
plt.show()

plt.figure()
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Charges")
plt.ylabel("Predicted Charges")
plt.title("Actual and Predicted Insurance Charges")
plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
)
plt.show()