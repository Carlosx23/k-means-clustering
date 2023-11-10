# Importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder

# Reading the dataset from a CSV file into a pandas DataFrame
df = pd.read_csv('Country.csv')  
print(df)

# Using LabelEncoder to convert 'child_mort' and 'gdpp' features into numerical values
LE = LabelEncoder()
df['child_mort'] = LE.fit_transform(df['child_mort'])
df['gdpp'] = LE.fit_transform(df['gdpp'])

# Selecting features for clustering and converting to NumPy array
x = np.asanyarray(df.drop(columns=['country']))

# Setting the number of clusters for K-Means
n = 4
model = KMeans(n_clusters=n)
model.fit(x)

# Predicting the cluster labels for each country
y = model.predict(x)

# Plotting the clusters in a 2D space
plt.figure()
plt.grid()
plt.xlabel('Feature 1 (child_mort)')
plt.ylabel('Feature 2 (gdpp)')
plt.title('K-Means Clustering for Countries')

# Plotting each cluster with a different color and printing country names in each cluster
for i in range(n):
    cluster_values = df.loc[y == i, 'country']
    plt.plot(x[y == i, 0], x[y == i, 1], 'o', label=f'Group {i + 1}')
    print(f"Countries with similar living conditions in Group {i+1}:")
    for name in cluster_values:
        print(name)

# Adding legend and displaying the plot
plt.legend()
plt.show()
