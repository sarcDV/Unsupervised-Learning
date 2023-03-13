"""
t-SNE stands for t-distributed stochastic neighbor embedding, 
which is a popular dimensionality reduction technique that converts similarities 
between data points to joint probabilities and tries to minimize the Kullback-Leibler 
divergence between the joint probabilities of the low-dimensional embedding and the high-dimensional data. 
"""

# Importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

# Loading the iris dataset
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
y = iris.target

# Standardizing the features
scaler = StandardScaler()
X = scaler.fit_transform(df)

# Applying t-SNE with 2 components and perplexity=30
tsne = TSNE(n_components=2, perplexity=30)
X_tsne = tsne.fit_transform(X)

# Plotting the results
plt.figure(figsize=(8,6))
plt.scatter(X_tsne[:,0], X_tsne[:,1], c=y, cmap='plasma')
plt.xlabel('First t-SNE component')
plt.ylabel('Second t-SNE component')
plt.show()
