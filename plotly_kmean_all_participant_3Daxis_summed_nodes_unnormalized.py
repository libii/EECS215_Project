# grab data
import csv

import matplotlib
from DataSet import DataSet #custom class
import numpy as np #scikit-learn requires this
import pandas as pd
import itertools

#kmeans
from sklearn.cluster import KMeans #sci-kit learn
import matplotlib.pyplot as plt # plotting
import plotly.express as px # interactive plotting
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

def elbow_method(data:np.ndarray):
    """Makes a plot using elbow methods for choosing clusters. Not as good as sillouete methods. But I did it because of a tuturial. I left it here but we will probably never use it."""
    inertia = []
    range_of_k = range(1, 11) # tries out different clusters from 1-10

    for k in range_of_k:
        kmeans = KMeans(n_clusters=k, random_state=33) # 42 is the seed because of Hitchhikers Guide. It's popularly used in tutorial. But I like 33 today
        kmeans.fit(data)
        inertia.append(kmeans.inertia_)

    plt.plot(range_of_k, inertia, marker='o')
    plt.title('The Elbow Method')
    plt.xlabel('# of Clusters')
    plt.ylabel('Inertia')
    plt.show() #### based on the graph 3 looks good. Because 3 has low enirtia 

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
        names.append(f'g{(i//4)+1}, {chr(96+p)}')
    return names

def stub(my_string:str="stub"):
    """function stub
       :param my_string: a string
       :returns: (str) a string"""
    return my_string

def main():
    num_groups=11
    total_participants=num_groups*4
    
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
        all_data[i][0]=prox_data.get_sum_all_nodes_normalize(2)[i]#/compelition[(i//4)][0]
        all_data[i][1]=convo_data.get_sum_all_nodes_normalize(2)[i]#/3)/compelition[(i//4)][0] # when it was summed it was just 3 values repeated
        all_data[i][2]=atten_data.get_sum_all_nodes_normalize(2)[i]#/compelition[(i//4)][0]

    # Kmeans Attempt
    ### Elbow Method #### - https://www.codecademy.com/learn/dspath-unsupervised/modules/dspath-clustering/cheatsheet (first figure)
    # elbow_method(all_data)

    ###### Make K Means Model and Extract Features ##########
    # Tell computer to divide in these number of clusters 
    num_clusters = 3 # used elbow method(data). for our data it was good at 3 n_clusters ... maybe 4 is better? Check the graph. I feel like it's a small change 3, 4.
    #number of clusters breaks 4 even though i want to try 4
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
        roles[label].append(f'g{(i//4)+1}, {chr(96+p)}')

    ### prints roles define by k cluster
    print(f'Role 1\tRole 2\tRole 3')#there is 3 if num_clusters=3
    print(2 * "_")

    for element in itertools.zip_longest(*roles):
        print(f'{element[0]}\t{element[1]}\t{element[2]}')#there is 3 if num_clusters=3

    benchmarks(kmeans=kmeans, num_clusters=num_clusters, data=data, labels=labels)
    ####### Plotting - 3 Features ######

    # Create DataFrame for plotly
    df = pd.DataFrame(data, columns=['Prox Count', 'Talking Duration', 'Shared Atten Count'])
    df['Cluster'] = [f'Cluster {i+1}' for i in labels]
    df['Label'] = get_names(num_groups=11)

    # Create 3D scatter plot
    fig = px.scatter_3d(
        df,
        x='Prox Count',
        y='Talking Duration',
        z='Shared Atten Count',
        color='Cluster',
        text='Label',
        title='KMeans Clustering in 3D'
    )

    # Add centroids
    fig.add_scatter3d(
        x=centroids[:,0],
        y=centroids[:,1],
        z=centroids[:,2],
        mode='markers',
        marker=dict(size=15, symbol='x', color='red'),
        name='Centroids'
    )

    # Update layout
    fig.update_traces(
        textposition='top center',
        textfont=dict(size=9)
    )
    fig.update_layout(
        scene=dict(
            xaxis_title='Prox Count',
            yaxis_title='Talking Duration', 
            zaxis_title='Shared Atten Count'
        ),
        legend_title_text='Clusters'
    )

    fig.show()


if __name__ == "__main__":
    main()