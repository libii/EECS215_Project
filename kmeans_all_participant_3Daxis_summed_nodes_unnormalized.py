# grab data
import csv

import matplotlib
from DataSet import DataSet #custom class
from OptimalClusterFinder import OptimalClusterFinder #custom class
import numpy as np #scikit-learn requires this
import itertools

#kmeans
from sklearn.cluster import KMeans #sci-kit learn
import matplotlib.pyplot as plt # plotting
from mpl_toolkits.mplot3d import Axes3D #3D MatPlotLib - if you have matplotlib, you have this

#benchmark tutorial
from time import time
from sklearn import metrics
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

#save
import os

def load_csv(file_name:str, directory:str="Data/")->list:
    """Load CSV from Data directory.
    :param file_name: Filename
    :param directory: Directory where file is stored.
    :returns: CSV data as a list contain a list for rows. Each row represents a group."""
    file_path="Data/"
    file_path+=file_name
    with open(file_path, 'r') as file:
        csv_reader = csv.reader(file)
        data = []
        for row in csv_reader:
            data.append(row)
    return data

def clean_compeletion_csv(data:list)->tuple:
    """Cleans up time/accuracy CSV for convient use.
        :param data: List of list from load_csv(file_name) where each row is a group.
        :returns: List of tuples (time (in secs):float, accuracy as percentage:float) with each row represents a group."""
    clean_data=[]
    data=data[1:]
    for row in data:
        junk, junk2, time = row[1].split()
        hours, minutes, seconds = map(float, time.split(':'))
        in_seconds = minutes * 60 + seconds
        clean_data.append((in_seconds,float(row[2])))
    return clean_data

    #k-means documention: https://scikit-learn.org/1.5/modules/generated/sklearn.cluster.KMeans.html
    #k-mean tutorial: https://scikit-learn.org/1.5/auto_examples/cluster/plot_kmeans_digits.html#sphx-glr-auto-examples-cluster-plot-kmeans-digits-py

def bench_k_means(kmeans, name, data, labels):
    """Benchmark to evaluate the KMeans initialization methods.

    Parameters
    ----------
    kmeans : KMeans instance
        A :class:`~sklearn.cluster.KMeans` instance with the initialization
        already set.
    name : str
        Name given to the strategy. It will be used to show the results in a
        table.
    data : ndarray of shape (n_samples, n_features)
        The data to cluster.
    labels : ndarray of shape (n_samples,)
        The labels used to compute the clustering metrics which requires some
        supervision.
    """
    t0 = time()
    estimator = make_pipeline(StandardScaler(), kmeans).fit(data)
    fit_time = time() - t0
    results = [name, fit_time, estimator[-1].inertia_]

    # Define the metrics which require only the true labels and estimator
    # labels
    clustering_metrics = [
        metrics.homogeneity_score,
        metrics.completeness_score,
        metrics.v_measure_score,
        metrics.adjusted_rand_score,
        metrics.adjusted_mutual_info_score,
    ]
    results += [m(labels, estimator[-1].labels_) for m in clustering_metrics]

    # The silhouette score requires the full dataset
    results += [
        metrics.silhouette_score(
            data,
            estimator[-1].labels_,
            metric="euclidean",
            sample_size=300,
        )
    ]

    # Show the results
    formatter_result = (
        "{:9s}\t{:.3f}s\t{:.0f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}"
    )
    print(formatter_result.format(*results))

def benchmarks(kmeans:str, num_clusters:int, data:np.ndarray, labels):
    """Benchmarks from an example kmean tutorial from scikit learn"""
    print(80 * "_")
    print("init\t\ttime\tinertia\thomo\tcompl\tv-meas\tARI\tAMI\tsilhouette")
    print(80 * "_")
    #evaluates our kmeans
    kmeans = KMeans(init="k-means++", n_clusters=num_clusters, n_init=4, random_state=0)
    bench_k_means(kmeans=kmeans, name="k-means++", data=data, labels=labels)

    #compares it to a randomly generated scatter graph with same sample size
    kmeans = KMeans(init="random", n_clusters=num_clusters, n_init=4, random_state=0)
    bench_k_means(kmeans=kmeans, name="random", data=data, labels=labels)

    pca = PCA(n_components=num_clusters).fit(data)
    kmeans = KMeans(init=pca.components_, n_clusters=num_clusters, n_init=1)
    bench_k_means(kmeans=kmeans, name="PCA-based", data=data, labels=labels)

    print(80 * "_")

def get_names(num_groups)->list:
    """Made this so I can add names to the dots in the matplotlib charts.
    :returns: array of strings with names of each person [g#, letter], where g# is the group number and letter is the person in the group (person a, person b, person c, person d)"""
    
    total_participants=num_groups
    names=[]
    p=None
    for i in range(num_groups*4):
        if i % 4 == 0:
            p=4
        else:
            p=i%4
        names.append(f'{(i//4)+1}{chr(96+p)}')
    return names

def main():
    num_groups=11
    total_participants=num_groups*4
    directory="/Graphs/All_Participant/Summed_Nodes/"
    graph_name="unnormalized_summed_nodes_of_for_3features"
    
    #loads csv data
    compelition=clean_compeletion_csv(load_csv("completion_time_and_accuracy.csv"))

    #load json data - must give a file name, can also take another folder relative to the location of the current file that calls it in the directory
    convo_data=DataSet("conversation_graphs.json")
    prox_data=DataSet("proximity_graphs.json")
    atten_data=DataSet("shared_attention_graphs.json")

    #numpy array where data set is column and nodes are rows
    data_sets=3 # different data sets, aka can be thought of as  features
    all_data=np.zeros((total_participants,data_sets))
    for i in range(total_participants):
        # these are counts or duration / seconds.##..
        all_data[i][0]=prox_data.get_sum_all_nodes()[i]/compelition[(i//4)][0]
        all_data[i][1]=(convo_data.get_sum_all_nodes()[i]/3)/compelition[(i//4)][0] # when it was summed it was just 3 values repeated
        all_data[i][2]=atten_data.get_sum_all_nodes()[i]/compelition[(i//4)][0]

   # determine # of clusters
    finder = OptimalClusterFinder(data=all_data, max_clusters=10, graph_name=graph_name,directory=directory)
    finder.find_optimal_clusters()
    optimal_clusters = finder.get_optimal_clusters()
    print(f"")
    finder.plot_combined_metrics()

    ###### Make K Means Model and Extract Features ##########
    # Tell computer to divide in these number of clusters 
    num_clusters = 4
    data=all_data

    # Create KMeans model and fit the data
    kmeans = KMeans(n_clusters=num_clusters, random_state=21) # seed at 21 because of forever 21
    kmeans.fit(data)

    # After the model is made, get the cluster centroids and labels
    centroids = kmeans.cluster_centers_ # the center points of the cluster generated by the Kmeans model for each feature
    labels = kmeans.labels_ # returns labels for each feature - this is useful because it tells us who is in what roles

    roles=[[] for _ in range(num_clusters)] # 3 if 3 labels, 4 if 4 labels. undecided
    p=None
    for i, label in enumerate(labels):
        if i % 4 == 0:
            p=4
        else:
            p=i%4
        roles[label].append(f'{(i//4)+1}{chr(96+p)}')

    ### prints roles define by k cluster
    print('\n'+graph_name)
    print(f'Role 1\tRole 2\tRole 3\tRole 4')#there is 3 if num_clusters=3
    print(29 * "_")

    for element in itertools.zip_longest(*roles):
        print(f'{element[0]}\t{element[1]}\t{element[2]}\t{element[3]}')#there is 3 if num_clusters=3

    # benchmarks(kmeans=kmeans, num_clusters=num_clusters, data=data, labels=labels)

    ####### Plotting - 3 Features ######
    # Plotting the results in 3D using axes 3d. Recommend matplotlib for 2d
    fig = plt.figure(figsize=(11, 7))
    gs = fig.add_gridspec(2, 3)
    ax = fig.add_subplot(gs[0, :], projection='3d')

    # Scatter plot
    for i in range(num_clusters):
        ax.scatter(data[labels == i, 0], data[labels == i, 1], data[labels == i, 2], label=f'Cluster {i + 1}')

    # Add labels to the dots
    name_labels=get_names(num_groups=11)
    for i in range(len(all_data)):
        ax.text(all_data[i][0],all_data[i][1],all_data[i][2], name_labels[i], fontsize=9)

    # Plot centroids - center dots for clusters
    ax.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2], s=350, c='red', marker='X', label='Centroids')

    ax.set_title('KMeans Clustering in 3D')
    ax.set_xlabel('Prox Count') # feature 1 - aka ndarray col 0
    ax.set_ylabel('Talking Duration') # feature 2 - aka ndarray col 1
    ax.set_zlabel('Shared Atten Count') # feature 3 - aka ndarray col 2
    ax.legend()

    ######## Plotting 2 feature of 3 feature Graph#####
    x_axis=1 #talking duration
    y_axis=2 #attention
    
    ax1 = fig.add_subplot(gs[1, 1])

    # Scatter plot
    for i in range(num_clusters):
        ax1.scatter(data[labels == i, x_axis], data[labels == i, y_axis], label=f'Cluster {i + 1}')

    # labels put in the plot
    for i in range(total_participants):
        ax1.text(data[i, x_axis], data[i, y_axis], name_labels[i])  # Label each point with its index

    # Plot centroids - center dots for clusters
    ax1.scatter(centroids[:, x_axis], centroids[:, y_axis], s=350, c='red', marker='X', label='Centroids')


    ax1.set_title("Talking and Attention")
    ax1.set_xlabel(f'Talking Duration')
    ax1.set_ylabel(f'Shared Attention')
    ax1.legend()

    ######## Plotting 2 feature of 3 feature Graph#####
    x_axis=0 #proximity
    y_axis=1 #talking
    
    ax2 = fig.add_subplot(gs[1, 0])

    # Scatter plot
    for i in range(num_clusters):
        ax2.scatter(data[labels == i, x_axis], data[labels == i, y_axis], label=f'Cluster {i + 1}')

    # labels put in the plot
    for i in range(total_participants):
        ax2.text(data[i, x_axis], data[i, y_axis], name_labels[i])  # Label each point with its index

    # Plot centroids - center dots for clusters
    ax2.scatter(centroids[:, x_axis], centroids[:, y_axis], s=350, c='red', marker='X', label='Centroids')

    ax2.set_title("Proximity and Talking")
    ax2.set_xlabel(f'Proximity')
    ax2.set_ylabel(f'Talking Duration')
    ax2.legend()

    x_axis=0 #Proximity
    y_axis=2 #attention
    
    ax3 = fig.add_subplot(gs[1, 2])

    # Scatter plot
    for i in range(num_clusters):
        ax3.scatter(data[labels == i, x_axis], data[labels == i, y_axis], label=f'Cluster {i + 1}')

    # labels put in the plot
    for i in range(total_participants):
        ax3.text(data[i, x_axis], data[i, y_axis], name_labels[i])  # Label each point with its index

    # Plot centroids - center dots for clusters
    ax3.scatter(centroids[:, x_axis], centroids[:, y_axis], s=350, c='red', marker='X', label='Centroids')

    ax3.set_title("Proximity and Attention")
    ax3.set_xlabel(f'Proximity')
    ax3.set_ylabel(f'Shared Attention')
    ax3.legend()

    #must save before show
    path=os.getcwd() 
    path += directory
    path += graph_name +".png"
    plt.savefig(path)

    # Add Verticle Space Padding
    plt.subplots_adjust(hspace=0.5)

    # Space padding around fig
    plt.tight_layout(pad=2.0)

    #show
    plt.show()

if __name__ == "__main__":
    main()