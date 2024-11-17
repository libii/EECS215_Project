import csv
from dataclasses    import dataclass
import dataclasses
from DataSet        import DataSet #custom class
from main           import load_csv, clean_compeletion_csv, map_label_participant, stub

import numpy as np

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

import matplotlib.pyplot as plt

import os #for save

class OptimalClusterFinder(object):
    def __init__(self, data, max_clusters=10, graph_name="default", directory="/"):
        """
            Initializes the OptimalClusterFinder class.
        
            Parameters:
            - data (numpy array or pandas DataFrame): The dataset to be clustered.
            - max_clusters (int): Maximum number of clusters to test.
        """
        self.data               = data
        self.max_clusters        = max_clusters
        self.inertia            = []
        self.silhouette_scores  = []
        self.graph_name=graph_name
        self.directory=directory

    def find_optimal_clusters(self):
        """
            Runs K-Means clustering for a range of clusters (from 2 to max_clusters),
            and records inertia and silhouette scores for each.
        """
        for n_clusters in range(2, self.max_clusters +1):
            kmeans = KMeans(n_clusters=n_clusters, random_state=86)
            kmeans.fit(self.data)

            #record the inertia (sum squared distances to closest cluster center)
            self.inertia.append(kmeans.inertia_)

            #Calculate and record the silhoutte score for current cluster count
            score = silhouette_score(self.data, kmeans.labels_)
            self.silhouette_scores.append(score)

    def plot_combined_metrics(self):
        """
            Plots the elbow curve (Inertia vs Number of clusters) to identify the optimal cluster count.
        """
        fig, ax1 = plt.subplots(figsize=(10,6))

        #plot inertia
        ax1.plot(range(2, self.max_clusters + 1), self.inertia, marker='o')
        ax1.set_xlabel("Number of Clusters")
        ax1.set_ylabel("Inertia", color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')

        #add secondary y-axis for silhouette scores 
        ax2 = ax1.twinx()
        ax2.plot(range(2, self.max_clusters + 1), self.silhouette_scores, marker='o', color='orange')
        ax2.set_ylabel("Silhouette Score", color='orange')
        ax2.tick_params(axis='y', labelcolor='orange')

        #Title and show the plot
        plt.title("Inertia and Silhouette Scores for optimal clustering")
        plt.tight_layout()
        self.save_graph()
        plt.show()

    def plot_elbow_method(self):
        """
        Plots the elbow curve (Inertia vs Number of clusters) to identify the optimal cluster count.
        """
        plt.figure(figsize=(10, 6))
        plt.plot(range(2, self.max_clusters + 1), self.inertia, marker='o')
        plt.title("Elbow Method for Optimal Number of Clusters")
        plt.xlabel("Number of Clusters")
        plt.ylabel("Inertia")
        plt.show()
    
    def plot_silhouette_scores(self):
        """
            Plots silhouette scores vs number of clusters to help determine the optimal number.
        """
        plt.figure(figsize=(10, 6))
        plt.plot(range(2, self.max_clusters + 1), self.silhouette_scores, marker='o', color='orange')
        plt.title("Silhouette Scores for Optimal Number of Clusters")
        plt.xlabel("Number of Clusters")
        plt.ylabel("Silhouette Score")
        plt.show()

    def get_optimal_clusters(self):
        """
            Determines and prints the optimal number of clusters based on silhouette scores.
        """
        optimal_clusters = np.argmax(self.silhouette_scores) + 2 #adjust for 0-based index
    
    def save_graph(self):
        """
            Saves graph in directory and names the file.
        """
        path=os.getcwd() 
        path += self.directory
        path += self.graph_name+"_sillhoutte_score" +".png"
        print(path)
        plt.savefig(path)



def main2():
    num_groups=11
    total_participants=num_groups*4

    #loads csv data
    # compelition=clean_compeletion_csv(load_csv("completion_time_and_accuracy.csv"))

    # # print time of first group - note: first in the tuple
    # print(f"Time of first Group:\t\t{compelition[0][0]}\n")
    # #print(compelition[0][0])

    # # print accuracy of first group - note: seconds in the tuple
    # print(f"accuracy of first Group:\t{compelition[0][1]}\n")
    # #print(compelition[0][1])

    #load json data - must give a file name, can also take another folder relative to the location of the current file that calls it in the directory
    convo_data=DataSet("conversation_graphs.json")
    prox_data=DataSet("proximity_graphs.json")
    atten_data=DataSet("shared_attention_graphs.json")

    # How to use Dataset
    # print json
    # prox_data.print_json() # it's long

    # print list of adj matrix
    # prox_data.print_adj_matrix() # it's long
    
    # Gets numpy ajacency matrix from a group
    # print(f"Numpy Adjacency Matrix from Group:\n{prox_data.get_group_matrix(2)}\n")

    # # calculate nodes of group and puts it in a list - order of participant: a, b, c, d
    # print(f"Calculates Nodes of group:\n{prox_data.get_sum_group_nodes(2)}\n")

    # # calculate all nodes of dataset in a list - order by group number (and participant is ordered is a, b, c, d)
    # print(f"Calculated nodes of dataset; ordered [a,b,c,d]:\n" +
    #     f"{prox_data.get_sum_all_nodes()}\n")

    #numpy array where data set is column and nodes are rows
    all_data=np.zeros((total_participants,3))
    for i in range(total_participants):
        all_data[i][0]=prox_data.get_sum_all_nodes()[i]
        all_data[i][1]=convo_data.get_sum_all_nodes()[i]
        all_data[i][2]=atten_data.get_sum_all_nodes()[i]

    # print(all_data) #it's long and gets converted to 64bit float which annoys me

    # To Think About: get some sort of average for conversation/proximity/shared attention divided by time from completion data (maybe I'm wrong)
    # To group members: to try stuff out on your own, maybe copy main.py as a templete since it loaded and transformt data
    #  rename your file so there aren't merge conflicts. Also, we should really talk to each other to figure out how to use kmeans

    # Eigenvalues
    # print(f"Eigenvalues:\n{np.linalg.eig(atten_data.list_adj_matrix[0]).eigenvalues}\n")

    # #Eigenvector
    # print(f"Eigenvectors:\n{np.linalg.eig(atten_data.list_adj_matrix[0]).eigenvectors}\n")

    # AV-Cluster finder:
    finder = OptimalClusterFinder(data=all_data, max_clusters=10)
    finder.find_optimal_clusters()
    optimal_clusters = finder.get_optimal_clusters()
    print(f"")
    finder.plot_combined_metrics()
  



if __name__ == "__main__":
    main2()

