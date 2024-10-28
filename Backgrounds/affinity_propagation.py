# sklearn affinity propagation example

from sklearn.datasets import make_blobs
from sklearn.cluster import AffinityPropagation
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score, silhouette_score
from itertools import cycle
import matplotlib.pyplot as plt
import numpy as np

def generate_sample_data():
    """Generate sample blob data for clustering"""
    centers = [[1, 1], [-1, -1], [1, -1]]
    return make_blobs(n_samples=300, centers=centers, 
                     cluster_std=0.5, random_state=0)

def fit_affinity_propagation(X):
    """Fit Affinity Propagation clustering model"""
    model = AffinityPropagation(preference=-50).fit(X)
    return (model.cluster_centers_indices_, 
            model.labels_, 
            len(model.cluster_centers_indices_))

def print_metrics(X, labels_true, labels):
    """Print clustering performance metrics"""
    print(f'Estimated number of clusters: {len(set(labels))}')
    print(f"Adjusted Rand Index: {adjusted_rand_score(labels_true, labels):.3f}")
    print(f"Adjusted Mutual Information: {adjusted_mutual_info_score(labels_true, labels):.3f}")
    print(f"Silhouette Coefficient: {silhouette_score(X, labels, metric='sqeuclidean'):.3f}")

def plot_clusters(X, labels, cluster_centers_indices):
    """Plot the clusters and their centers"""
    colors = cycle('rgb')
    for k, col in zip(range(len(cluster_centers_indices)), colors):
        class_members = labels == k
        cluster_center = X[cluster_centers_indices[k]]
        
        # Plot points
        plt.plot(X[class_members, 0], X[class_members, 1], col + '.')
        
        # Plot lines connecting points to centers
        for x in X[class_members]:
            plt.plot([cluster_center[0], x[0]], 
                    [cluster_center[1], x[1]], 
                    col, alpha=0.25)
        
        # Plot cluster centers
        plt.plot(cluster_center[0], cluster_center[1], 
                'o', mec='k', mew=3, markersize=7)

def main():
    # Generate and fit data
    X, labels_true = generate_sample_data()
    cluster_centers_indices, labels, n_clusters = fit_affinity_propagation(X)
    
    # Print metrics and plot results
    print_metrics(X, labels_true, labels)
    plot_clusters(X, labels, cluster_centers_indices)
    plt.show()

if __name__ == "__main__":
    main()