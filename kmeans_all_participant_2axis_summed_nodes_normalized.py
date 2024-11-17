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
from sklearn.metrics import silhouette_samples

#benchmark tutorial
from time import time
from sklearn import metrics
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

#save
import os

#debugging
from pprint import pprint 

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
    graph_name="new_normalized_summed_nodes_of_for_2features"
    
    #load json data - must give a file name, can also take another folder relative to the location of the current file that calls it in the directory
    prox_data=DataSet("proximity_graphs.json")
    convo_data=DataSet("conversation_graphs.json")
    atten_data=DataSet("shared_attention_graphs.json")

    fig = plt.figure(figsize=(15, 5))
    data_sets=2
    num_clusters = 3
    data_features=np.zeros((total_participants,data_sets))

    #Overhead: so I can do this in one snazzy loop
    features=[prox_data, convo_data, atten_data]
    axises=[(0,1),(0,2),(1,2)]
    axis_name=["Proximity", "Talking", "Attention"]
    x_axis=0 
    y_axis=1 
    name_labels=get_names(num_groups=11)
    no_norm_data_features=np.zeros((total_participants,data_sets))


    #Doing 2 Feature Kmeans for 3 combos of graphs
    for index, axis in enumerate(axises):
        for i in range(total_participants):
            data_features[i][0]=features[axis[0]].get_sum_all_nodes_normalize(1)[i]
            data_features[i][1]=features[axis[1]].get_sum_all_nodes_normalize(1)[i]

        ###### Make K Means Model and Extract Features ##########
        data=data_features

        # determine # of clusters
        # leave this commented to show the final graph, you can view 3 silloute score OR final graph
        # this_graph_name=str(graph_name + "_" + axis_name[axis[0]] + "_" + axis_name[axis[1]])
        # finder = OptimalClusterFinder(data=data, max_clusters=10, graph_name=this_graph_name,directory=directory)
        # finder.find_optimal_clusters()
        # optimal_clusters = finder.get_optimal_clusters()
        # print(f"")
        # finder.plot_combined_metrics()

        # if axis==(0,1): #prox/convo
        #     num_clusters=3
        # elif axis==(0,2): #prox/attention
        #     num_clusters=3
        # elif axis==(1,2): #convo/atten
        #     num_clusters=3

        # Create KMeans model and fit the data
        kmeans = KMeans(n_clusters=num_clusters, random_state=21) # seed at 21 because of forever 21
        kmeans.fit(data)

        # After the model is made, get the cluster centroids and labels
        centroids = kmeans.cluster_centers_ # the center points of the cluster generated by the Kmeans model for each feature
        labels = kmeans.labels_ # returns labels for each feature - this is useful because it tells us who is in what roles

        # Make silhouette scores core each person
        silhouette_scores = silhouette_samples(data, labels)

        # Print silhouette scores
        # for i, score in enumerate(silhouette_scores):
        #     if i % 4 == 0:
        #         p=4
        #     else:
        #         p=i%4
        #     print(f"Point {(i//4)+1}{chr(96+p)}: Silhouette Score = {score}")

        roles=[[] for _ in range(num_clusters)]
        p=None

        for i, (label, score) in enumerate(zip(labels, silhouette_scores)):
            if i % 4 == 0:
                p=4
            else:
                p=i%4
            roles[label].append(f'{(i//4)+1}{chr(96+p)}{score:.2f}')

        ## prints roles define by k cluster
        print_name=graph_name + ":\n"+ axis_name[axis[0]] +" & "+axis_name[axis[1]]
        print(print_name)

        print(f'Role 1\tRole 2\tRole 3')#there is 3 if num_clusters=3
        print(22 * "_")

        for element in itertools.zip_longest(*roles):
            print(f'{element[0]}\t{element[1]}\t{element[2]}')#there is 3 if num_clusters=3

        print("\n")

        ######## Plotting 2 feature of 3 feature Graph ########

        num=130+1+index
        ax1 = fig.add_subplot(num)

        # Scatter plot
        for i in range(num_clusters):
            ax1.scatter(data[labels == i, x_axis], data[labels == i, y_axis], label=f'Cluster {i + 1}')

        # labels put in the plot
        for i in range(total_participants):
            ax1.text(data[i, x_axis], data[i, y_axis], name_labels[i])  # Label each point with its index

        # Plot centroids - center dots for clusters
        ax1.scatter(centroids[:, x_axis], centroids[:, y_axis], s=350, c='red', marker='X', label='Centroids')

        ax1.set_title(f'Kmeans for 2 Feature - {axis_name[axis[0]]} and {axis_name[axis[1]]}')
        ax1.set_xlabel(f'{axis_name[axis[0]]}')
        ax1.set_ylabel(f'{axis_name[axis[1]]}')
        ax1.legend()


    path=os.getcwd() 
    path += directory
    path += graph_name +".png"
    plt.savefig(path)

    plt.show()

if __name__ == "__main__":
    main()