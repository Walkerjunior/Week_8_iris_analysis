# iris_analysis.py

# Task: Data Loading and Exploration

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

# Error handling for data loading
try:
    # Load the Iris dataset
    iris_data = load_iris()
    df = pd.DataFrame(data=iris_data.data, columns=iris_data.feature_names)
    df['species'] = pd.Categorical.from_codes(iris_data.target, iris_data.target_names)
    print("Dataset loaded successfully.")
except Exception as e:
    print("Error loading dataset:", e)

# Display first few rows
print("\nFirst 5 rows of the dataset:")
print(df.head())

# Explore structure
print("\nData Types:")
print(df.dtypes)

print("\nMissing Values:")
print(df.isnull().sum())

# No missing values in this dataset

# Task: Basic Data Analysis
print("\nDescriptive Statistics:")
print(df.describe())

# Grouping by species and computing the mean
print("\nMean values by species:")
print(df.groupby('species').mean())

# Observations
print("\nObservations:")
print("- Setosa has the smallest petal sizes on average.")
print("- Virginica has the largest sepal length and petal length.")
print("- Versicolor is in between for most measurements.")

# Task: Data Visualization

# Set style for seaborn
sns.set(style="whitegrid")

# 1. Line chart: Simulating a trend (not time-series, but example using index)
plt.figure(figsize=(8, 5))
df_sorted = df.sort_values(by='sepal length (cm)')
plt.plot(df_sorted.index, df_sorted['sepal length (cm)'], label='Sepal Length')
plt.title('Sepal Length Trend (Sorted by Length)')
plt.xlabel('Index')
plt.ylabel('Sepal Length (cm)')
plt.legend()
plt.tight_layout()
plt.show()

# 2. Bar chart: Average petal length by species
plt.figure(figsize=(6, 4))
sns.barplot(x='species', y='petal length (cm)', data=df, ci=None)
plt.title('Average Petal Length by Species')
plt.xlabel('Species')
plt.ylabel('Petal Length (cm)')
plt.tight_layout()
plt.show()

# 3. Histogram: Distribution of sepal width
plt.figure(figsize=(6, 4))
plt.hist(df['sepal width (cm)'], bins=15, color='skyblue', edgecolor='black')
plt.title('Distribution of Sepal Width')
plt.xlabel('Sepal Width (cm)')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()

# 4. Scatter plot: Sepal Length vs. Petal Length
plt.figure(figsize=(6, 4))
sns.scatterplot(x='sepal length (cm)', y='petal length (cm)', hue='species', data=df)
plt.title('Sepal Length vs Petal Length')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Petal Length (cm)')
plt.tight_layout()
plt.show()
