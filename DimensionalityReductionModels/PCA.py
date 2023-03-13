"""
PCA stands for Principal Component Analysis, which is a multivariate statistical technique that 
transforms the data from high dimension space to low dimension space with minimal loss 
of information and also removing the redundancy in the dataset1. 
"""
# Importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Loading the iris dataset
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
y = iris.target

# Standardizing the features
scaler = StandardScaler()
X = scaler.fit_transform(df)

# Applying PCA with 2 components
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Plotting the results
plt.figure(figsize=(8,6))
plt.scatter(X_pca[:,0], X_pca[:,1], c=y, cmap='plasma')
plt.xlabel('First principal component')
plt.ylabel('Second principal component')
plt.show()
