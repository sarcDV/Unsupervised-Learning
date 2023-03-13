# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Generating some random data
X = np.random.randn(1000, 2)

# Plotting the data
plt.scatter(X[:, 0], X[:, 1])
plt.title('Random data')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

# Applying k-means with 3 clusters
kmeans = KMeans(n_clusters=3, random_state=0)
kmeans.fit(X)

# Getting the cluster labels and centroids
labels = kmeans.labels_
centroids = kmeans.cluster_centers_

# Plotting the clusters and centroids
plt.scatter(X[:, 0], X[:, 1], c=labels)
plt.scatter(centroids[:, 0], centroids[:, 1], marker='*', s=200, c='red')
plt.title('K-means clustering')
plt.xlabel('x')
plt.ylabel('y')
plt.show()
##### example using pytorch library
# Importing the libraries
import torch
import numpy as np
from kmeans_pytorch import kmeans # This is a library that implements k-means for pytorch

# Generating some random data
data_size, dims, num_clusters = 1000, 2, 3
x = np.random.randn(data_size, dims) / 6
x = torch.from_numpy(x)

# Applying k-means with 3 clusters
cluster_ids_x, cluster_centers = kmeans(
    X=x, num_clusters=num_clusters, distance='euclidean', device=torch.device('cuda:0')
)

# Plotting the clusters and centroids
import matplotlib.pyplot as plt
plt.scatter(x[:, 0], x[:, 1], c=cluster_ids_x)
plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], marker='*', s=200, c='red')
plt.title('K-means clustering')
plt.xlabel('x')
plt.ylabel('y')
plt.show()
