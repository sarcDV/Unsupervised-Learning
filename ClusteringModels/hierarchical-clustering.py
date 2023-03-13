# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets import make_blobs

# Generating some blob-shaped data
X, y = make_blobs(n_samples=1000, centers=4, cluster_std=0.5, random_state=0)

# Plotting the data
plt.scatter(X[:, 0], X[:, 1])
plt.title('Blob-shaped data')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

# Applying AgglomerativeClustering with n_clusters=4 and linkage='ward'
agg = AgglomerativeClustering(n_clusters=4, linkage='ward')
agg.fit(X)

# Getting the cluster labels
labels = agg.labels_

# Plotting the clusters
plt.scatter(X[:, 0], X[:, 1], c=labels)
plt.title('AgglomerativeClustering with n_clusters=4 and linkage=ward')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

"""
Hierarchical clustering is a method of cluster analysis that seeks to build a hierarchy of clusters. 
It can be done in two ways: agglomerative or divisive. Agglomerative hierarchical clustering works 
by sequentially merging similar clusters, starting from individual data points and ending with a 
single cluster containing all the data points. Divisive hierarchical clustering works by 
initially grouping all the data points into one cluster, and then successively splitting these 
clusters into smaller ones until each data point is in its own cluster.

The similarity or distance between clusters can be measured in different ways, such as Euclidean distance, 
Manhattan distance, cosine similarity, etc. The linkage criterion determines how the distance between clusters is calculated. 
Some common linkage criteria are single linkage (minimum distance), complete linkage (maximum distance), 
average linkage (average distance), and ward linkage (minimum variance).

Hierarchical clustering has some advantages over other clustering algorithms, such as being able to 
find clusters of arbitrary shape, being able to visualize the clustering process using a dendrogram, 
and not requiring to specify the number of clusters in advance. However, it also has some drawbacks, 
such as being sensitive to the choice of distance measure and linkage criterion, 
having difficulty with outliers and noisy data, and being computationally expensive on large datasets.
"""
