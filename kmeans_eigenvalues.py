from DataSet        import DataSet #custom class
from main           import load_csv, clean_compeletion_csv, map_label_participant, stub
from OptimalClusterFinder import OptimalClusterFinder

import numpy  as np
import matplotlib.pyplot as plt

class kmeans_eigenvalues(object):
	def __init__(self, list_adj_matrix:list):
		"""
			Initializes the kmeans_eigenvalues class.
        
			Parameters:
			- list_adj_matrix (list of np.arrays): The list of adj matrixes to be converted to eigenvalues
		"""
		self.list_adj_matrix	= list_adj_matrix
		self.eigenvalues		= []
		self.list_eigenvalues	= self.adj_list_to_eigenvalues()


	def adj_list_to_eigenvalues(self):
		"""
			Convert adjacency list to eigenvalue array
		"""
		for matrix in self.list_adj_matrix:
			if matrix.shape != (4,4):
				raise ValueError("all matrices are expected to be 4x4")
			eigenvalues = np.linalg.eigvals(matrix)
			self.eigenvalues.extend(eigenvalues) # need to use extend so that everything is in the same list

		
	
if __name__ == "__main__":
	print("getting eigenvalues")
	
	#initialize data
	compelition	= clean_compeletion_csv(load_csv("completion_time_and_accuracy.csv"))
	convo_data	= DataSet("conversation_graphs.json")
	prox_data	= DataSet("proximity_graphs.json")
	atten_data	= DataSet("shared_attention_graphs.json")

	#convert ot eigen values
	convo_data_eigen	= kmeans_eigenvalues(convo_data.list_adj_matrix).eigenvalues
	prox_data_eigen		= kmeans_eigenvalues(prox_data.list_adj_matrix).eigenvalues
	atten_data_eigen	= kmeans_eigenvalues(atten_data.list_adj_matrix).eigenvalues

	#get number of participants
	num_participants = len(convo_data_eigen)

	#consolidate all data
	all_eigen_data = np.zeros((num_participants,3))
	for i in range(num_participants):
		all_eigen_data[i][0] = convo_data_eigen[i]
		all_eigen_data[i][1] = prox_data_eigen[i]
		all_eigen_data[i][2] = atten_data_eigen[i]
	
	finder = OptimalClusterFinder(data=all_eigen_data, max_clusters=10)
	finder.find_optimal_clusters()
	finder.plot_combined_metrics()


