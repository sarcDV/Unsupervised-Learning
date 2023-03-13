# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_moons

# Generating some moon-shaped data
X, y = make_moons(n_samples=1000, noise=0.05)

# Plotting the data
plt.scatter(X[:, 0], X[:, 1])
plt.title('Moon-shaped data')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

# Applying DBSCAN with eps=0.3 and min_samples=5
dbscan = DBSCAN(eps=0.3, min_samples=5)
dbscan.fit(X)

# Getting the cluster labels and core samples
labels = dbscan.labels_
core_samples_mask = np.zeros_like(labels, dtype=bool)
core_samples_mask[dbscan.core_sample_indices_] = True

# Plotting the clusters and outliers
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_outliers = list(labels).count(-1)
unique_labels = set(labels)
colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black used for outliers.
        col = [0, 0, 0, 1]

    class_member_mask = (labels == k)

    xy = X[class_member_mask & core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=6)

    xy = X[class_member_mask & ~core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=3)

plt.title(f'DBSCAN clustering\nEstimated number of clusters: {n_clusters}\nEstimated number of outliers: {n_outliers}')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

"""
DBSCAN is a density-based clustering algorithm that works on the assumption 
that clusters are dense regions in space separated by regions of lower density. 
It can identify clusters in large spatial datasets by looking at the local density of the data points.

DBSCAN creates a circle of epsilon radius around every data point and classifies them into Core point, 
Border point, and Noise. A data point is a Core point if the circle around it contains at least ‘minPoints’ 
number of points. A data point is a Border point if it is not a Core point but is reachable from a Core point. 
A data point is a Noise point if it is neither a Core point nor a Border point.

DBSCAN then forms clusters by connecting Core points that are close to each other 
(within epsilon distance), and adding Border points that are close to a Core point to the same cluster. 
Noise points are not assigned to any cluster.

DBSCAN has some advantages over other clustering algorithms, such as being able to find 
clusters of arbitrary shape, being robust to outliers, and not requiring to specify the number 
of clusters in advance. However, it also has some drawbacks, such as being sensitive to the choice of 
epsilon and minPoints parameters, having difficulty with varying densities, and being slow on large datasets.
"""
