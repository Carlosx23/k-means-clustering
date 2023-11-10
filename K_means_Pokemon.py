# Importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder

# Reading the Pokémon dataset from a CSV file into a pandas DataFrame
df = pd.read_csv('Pokemon.csv')  
print(df)

# Using LabelEncoder to convert categorical columns 'Type_1' and 'Legendary' into numerical values
LE = LabelEncoder()
df['Type_1'] = LE.fit_transform(df['Type_1'])
df['Legendary'] = LE.fit_transform(df['Legendary'])

# Selecting features for clustering and converting to NumPy array
x = np.asanyarray(df.drop(columns=['Name']))

# Setting the number of clusters (groups) for K-Means
n = 4
model = KMeans(n_clusters=n)
model.fit(x)

# Predicting the cluster labels for each Pokémon
y = model.predict(x)

# Plotting the clusters in a 2D space
plt.figure()
plt.grid()
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('K-Means Clustering')

# Plotting each cluster with a different color and printing Pokémon names in each cluster
for i in range(n):
    cluster_values = df.loc[y == i, 'Name']
    plt.plot(x[y == i, 0], x[y == i, 1], 'o', label=f'Group {i + 1}')
    print(f"Pokémon with similar characteristics in Group {i+1}:")
    for name in cluster_values:
        print(name)

# Adding legend and displaying the plot
plt.legend()
plt.show()
