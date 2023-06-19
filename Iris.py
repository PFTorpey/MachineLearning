
'''
Unsupervised machine learning to analyze the iris dataset.
Methods used:
1. K-means Clustering
2. Density-based Clustering Algorithm (DBSCAN)
3. Principal Component Analysis (PCA)
'''


import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Load the Iris dataset
iris = load_iris()
data = pd.DataFrame(data=iris.data, columns=iris.feature_names)

# Separate the features
X = data.values

# Create K-means, DBSCAN, and GMM models
kmeans_model = KMeans(n_clusters=3)
dbscan_model = DBSCAN(eps=0.5, min_samples=5)
gmm_model = GaussianMixture(n_components=3)

# Fit the models and obtain cluster labels
kmeans_labels = kmeans_model.fit_predict(X)
dbscan_labels = dbscan_model.fit_predict(X)
gmm_labels = gmm_model.fit(X).predict(X)

# Apply PCA for dimensionality reduction and visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Visualize the results
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=kmeans_labels, cmap='viridis')
plt.title('K-means Clustering')

plt.subplot(1, 3, 2)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=dbscan_labels, cmap='viridis')
plt.title('DBSCAN')

plt.subplot(1, 3, 3)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=gmm_labels, cmap='viridis')
plt.title('Gaussian Mixture Model')

plt.tight_layout()
plt.show()