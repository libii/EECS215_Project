# python undirected graph example

from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np

def generate_sample_data():
    """Generate sample blob data for clustering"""
    centers = [[1, 1], [-1, -1], [1, -1]]
    return make_blobs(n_samples=300, centers=centers,
                     cluster_std=0.5, random_state=0)

def fit_kmeans(X, n_clusters=3):
    """Fit K-means clustering model"""
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    kmeans.fit(X)
    return kmeans.labels_, kmeans.cluster_centers_

def plot_clusters(X, labels, centers):
    """Plot the clusters and their centers"""
    # Create scatter plot of points colored by cluster
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
    
    # Plot the cluster centers
    plt.scatter(centers[:, 0], centers[:, 1], 
               c='red', marker='x', s=200, linewidth=3,
               label='Centroids')
    
    plt.title('K-means Clustering')
    plt.legend()

def main():
    # Generate sample data
    X, _ = generate_sample_data()
    
    # Fit K-means and get labels and centers
    labels, centers = fit_kmeans(X)
    
    # Plot results
    plot_clusters(X, labels, centers)
    plt.show()

if __name__ == "__main__":
    main()
