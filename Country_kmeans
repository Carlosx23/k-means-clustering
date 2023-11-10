# Importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder

# Reading the dataset from a CSV file into a pandas DataFrame
df = pd.read_csv('crime_data.csv')  
print(df)

# Using LabelEncoder to convert 'Murder' and 'Rape' features into numerical values
LE = LabelEncoder()
df['Murder'] = LE.fit_transform(df['Murder'])
df['Rape'] = LE.fit_transform(df['Rape'])

# Selecting features for clustering and converting to NumPy array
x = np.asanyarray(df.drop(columns=['Target']))

# Setting the number of clusters for K-Means
n = 4
model = KMeans(n_clusters=n)
model.fit(x)

# Predicting the cluster labels for each state
y = model.predict(x)

# Plotting the clusters in a 2D space
plt.figure()
plt.grid()
plt.xlabel('Feature 1 (Murder)')
plt.ylabel('Feature 2 (Rape)')
plt.title('K-Means Clustering for Crime Data')

# Plotting each cluster with a different color and printing state names in each cluster
for i in range(n):
    cluster_values = df.loc[y == i, 'Target']
    plt.plot(x[y == i, 0], x[y == i, 1], 'o', label=f'Group {i + 1}')
    print(f"States with similar crime levels in Group {i+1}:")
    for name in cluster_values:
        print(name)

# Adding legend and displaying the plot
plt.legend()
plt.show()
