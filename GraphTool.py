from DataSet        import DataSet #custom class
from main           import load_csv, clean_compeletion_csv, map_label_participant, stub
from kmeans_eigenvalues		import kmeans_eigenvalues
from OptimalClusterFinder	import OptimalClusterFinder
from kmeans_all_participant_3Daxis_summed_nodes_normalized import main as all_l2Norm_main, get_names, benchmarks, bench_k_means

#kmeans
from sklearn.cluster import KMeans #sci-kit learn
from mpl_toolkits.mplot3d import Axes3D #3D MatPlotLib - if you have matplotlib, you have this
from sklearn.metrics import silhouette_samples

#benchmark tutorial
from time import time
from sklearn import metrics
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


import numpy				as np
import matplotlib.pyplot	as plt
import itertools



class GraphTool(object):
	def __init__(self,all_data,n_clusters,num_groups=11):
		self.all_data		=  all_data
		self.name_labels	= get_names(num_groups)
		self.n_clusters		= n_clusters
		self.kmeans			= None
		self.labels			= None
		self.centroids		= None
		self.fit()
		self.group_size		= 4
		self.role_dict		= self.get_roles()
		self.grid_mapping	= self.map_to_grid()

	def fit(self):
		"""
			Fit the Kmeans model to the input data
		"""
		self.kmeans		= KMeans(n_clusters=self.n_clusters, random_state=21) # 21 forever lol
		self.kmeans.fit(self.all_data)
		self.labels		= self.kmeans.labels_
		self.centroids	= self.kmeans.cluster_centers_

	def plot_roles(self, role_colors=None):
		"""
			WIP = work in progress
			visualize the grouroles
		"""
		if self.labels is None or self.centroids is None:
			raise ValueError("Model has not been fitted. Call the 'fit' method first.")

		#default colors if none are provided
		if role_colors is None:
			cmap = plt.get_cmap("tab10") # up to ten unique colors
			role_colors = {i: cmap(i) for i in range(self.n_clusters)}

		#create grid
		fig, ax = plt.subplots(figsize=(8, 6))
		for role, positions in self.grid_mapping.items():
			color = role_colors.get(role, "gray")  # Default to gray if role not found
			for row, col in positions:
				# Adjust row for visualization (matplotlib uses bottom-left as origin)
				adjusted_row = row - 1
				ax.add_patch(plt.Rectangle((col - 0.5, adjusted_row - 0.5), 1, 1, color=color, edgecolor="black"))
				ax.text(col, adjusted_row, role, ha="center", va="center", fontsize=10, color="white")

		# Set grid limits
		max_row = max(pos[0] for positions in self.grid_mapping.values() for pos in positions)
		max_col = max(pos[1] for positions in self.grid_mapping.values() for pos in positions)
		ax.set_xlim(0.5, max_col + 0.5)
		ax.set_ylim(-0.5, max_row - 0.5)
		ax.set_xticks(range(1, max_col + 1))
		ax.set_yticks(range(max_row))
		ax.invert_yaxis()  # Invert y-axis to make row 1 appear at the top
		plt.grid(False)
		plt.title("Roles among groups")
		plt.xlabel("Individual")
		plt.ylabel("Groups")
		plt.show()


	def get_roles(self) -> dict:
		"""
			Mapping the names of each label with their role from the kmeans
		"""
		print("Get Roles:\n")
		roles	= {name: cluster for name, cluster in zip(self.name_labels, self.labels)}
		return roles

	def map_to_grid(self):
		"""
			Maps dictionary values like '8a' to grid positions.
			Row is determined by the digit, and column by the letter's alphabetical position.

			:param role_dict: Dictionary of roles and their positions (e.g., {'role1': '8a', ...}).
			:return: Dictionary mapping roles to grid positions (row, col).
		"""
		grid_mapping = {}
		
		for group, role in self.role_dict.items():
			#get row
			row = int(group[:-1]) #get all characters but the last one
			#get col
			col = ord(group[-1].lower()) - ord('a') + 1 #get last entry and convert to number a=1, etc...

			if role not in grid_mapping:
				grid_mapping[role] = [] #initialize role grid
			grid_mapping[role].append((row,col))
			
		return grid_mapping


def main_graph_tool():
	print("getting Graphs")
	
	#data initial parameters
	num_groups = 11
	data_sets = 4
	total_participants = num_groups*4
	
	
	#kmeans parameters
	num_clusters = 3

	#initialize data
	compelition	= clean_compeletion_csv(load_csv("completion_time_and_accuracy.csv"))
	convo_data	= DataSet("conversation_graphs.json")
	prox_data	= DataSet("proximity_graphs.json")
	atten_data	= DataSet("shared_attention_graphs.json")

	#convert to eigen values
	convo_data_eigen	= kmeans_eigenvalues(convo_data.list_adj_matrix).eigenvalues
	prox_data_eigen		= kmeans_eigenvalues(prox_data.list_adj_matrix).eigenvalues
	atten_data_eigen	= kmeans_eigenvalues(atten_data.list_adj_matrix).eigenvalues
	
	#convert to Normalized values
	convo_data_norm		= convo_data.get_sum_all_nodes_normalize(2)
	prox_data_norm		= prox_data.get_sum_all_nodes_normalize(2)
	atten_data_norm		= atten_data.get_sum_all_nodes_normalize(2)

	#consolidate all eigenvalue data
	num_participants = len(convo_data_eigen)
	all_eigen_data = np.zeros((num_participants,3))
	for i in range(num_participants):
		all_eigen_data[i][0] = convo_data_eigen[i]
		all_eigen_data[i][1] = prox_data_eigen[i]
		all_eigen_data[i][2] = atten_data_eigen[i]

	#consolidate all normalized data
	num_participants = len(convo_data_norm)
	all_data = np.zeros((total_participants, data_sets))
	for i in range(num_participants):
		all_data[i][0] = convo_data_norm[i]
		all_data[i][1] = prox_data_norm[i]
		all_data[i][2] = atten_data_norm[i]
	
	visualizer = GraphTool(all_data,n_clusters=3,num_groups=11)
	visualizer.plot_roles()


if __name__ == "__main__":
	main_graph_tool()
