import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import SpectralClustering
from sklearn.datasets import make_moons

# Generate synthetic data
X, y = make_moons(n_samples=300, noise=0.05, random_state=42)

# Apply Spectral Clustering
n_clusters = 2  # Number of clusters
spectral = SpectralClustering(n_clusters=n_clusters, affinity='nearest_neighbors', random_state=42)
labels = spectral.fit_predict(X)
print(f"labels: {labels}")

# Plot clustered data
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=50)
plt.title("Spectral Clustering Results")
plt.show()