import pandas as pd
from sklearn.datasets import load_iris

iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['target'] = iris.target
print(df.head())
print(df.shape)
print(df.columns)
print(df.info())
print(df.describe())

#Histogram
import matplotlib.pyplot as plt
df.hist(figsize=(10, 8))
plt.show()

#Scatter Plot
plt.scatter(df['sepal length (cm)'], df['sepal width (cm)'])
plt.xlabel('sepal length (cm)')
plt.ylabel('sepal width (cm)')
plt.show()

#Seaborn
import seaborn as sns
sns.pairplot(df, hue='target')
plt.show()
