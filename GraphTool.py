from statistics import LinearRegression
from DataSet        import DataSet #custom class
from main           import load_csv, clean_compeletion_csv, map_label_participant, stub
from kmeans_eigenvalues		import kmeans_eigenvalues
from OptimalClusterFinder	import OptimalClusterFinder
from kmeans_all_participant_3Daxis_summed_nodes_normalized import main as all_l2Norm_main, get_names, benchmarks, bench_k_means

#kmeans
from sklearn.cluster		import KMeans #sci-kit learn

#benchmark tutorial
from sklearn.pipeline		import make_pipeline
from sklearn.preprocessing	import StandardScaler
from sklearn.cluster		import KMeans
from sklearn.decomposition	import PCA
from sklearn.linear_model	import LinearRegression
from sklearn.metrics		import r2_score


import numpy				as np
import matplotlib.pyplot	as plt



class RoleGraph(object):
	def __init__(self,all_data,n_clusters, data_accuracy,num_groups=11):
		self.all_data		=  all_data
		self.time, self.accuracy = zip(*data_accuracy) #unpack time and accuracy
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

		#add in accuracy Col
		for i in range(0, len(self.accuracy)):
			ax.add_patch(plt.Rectangle((5 - 0.5, i - 0.5), 1, 1, color="purple", edgecolor="black"))
			ax.text(5, i, round(self.accuracy[i],2) , ha="center", va="center", fontsize=10, color="white")

		#add in Time column
		for i in range(0, len(self.time)):
			ax.add_patch(plt.Rectangle((6 - 0.5, i - 0.5), 1, 1, color="red", edgecolor="black"))
			ax.text(6, i, round(self.time[i]/60,2) , ha="center", va="center", fontsize=10, color="white")

		# Set grid limits
		max_row = max(pos[0] for positions in self.grid_mapping.values() for pos in positions) 
		max_col = max(pos[1] for positions in self.grid_mapping.values() for pos in positions) + 1
		ax.set_xlim(0.5, max_col + 0.5 + 1)
		ax.set_ylim(-0.5, max_row - 0.5)
		ax.set_xticks(range(1, max_col + 1+1))
		ax.set_yticks(range(1, max_row))
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

	def map_to_grid(self) -> dict:
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


class ScatterMetricVsAccuracy:
	def __init__(self, data, data_accuracy):
		"""
			Initialize the class with data and accuracy values.
			:param data: A 2D NumPy array where each row represents a data point.
			:param accuracy_data: A list of tuples where each tuple is (time, accuracy).
        """
		self.data = data
		self.time, self.accuracy = zip(*data_accuracy) #unpack time and accuracy
		self.model = None

	def regression_plot(self):
		"""
		Create a scatter plot with a regression plot
		"""
		num_columns = self.data.shape[1]
		colors = ['blue', 'green', 'orange', 'purple', 'red']  # Define colors for each column
		plt.figure(figsize=(10, 6))

		for i in range(num_columns):
			# Extract column data
			x = self.data[:, i]
			y = self.time
			data_labels = ['Conversation', 'Proximity', 'Attention']

			# Scatter plot for this column
			plt.scatter(x, y, color=colors[i % len(colors)], label=f"{data_labels[i]} Data")

			# Fit a regression line
			model = LinearRegression()
			x_reshaped = x.reshape(-1, 1)
			model.fit(x_reshaped, y)
			predicted = model.predict(x_reshaped)			

			# Plot regression line
			plt.plot(x, predicted, color=colors[i % len(colors)], linestyle='--', label=f"{data_labels[i]} Regression\n$R^2 = {r2_score(y, predicted):.2f}$")

			# Add regression equation
			intercept = model.intercept_
			slope = model.coef_[0]
			equation_text = f"$y = {slope:.2f}x + {intercept:.2f}$"
			plt.text(5, np.max(y) - i * 2, equation_text, fontsize=10, color=colors[i % len(colors)])

		# Labels, legend, and grid
		plt.xlabel("Eigen Value Engergy of Group")
		plt.ylabel("Time (sec)")
		plt.title("Eigen Values vs Group Time")
		plt.legend()
		#plt.legend(loc="center left", bbox_to_anchor=(0, 0.2))
		plt.grid(True)
		plt.show()

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
	

	# row is person, col is data sets
	group_energy_data=np.zeros((num_groups, 3))
	for i in range(num_groups):
		group_energy_data[i][0]=prox_data.get_group_energy_laplacian(i+1)
		group_energy_data[i][1]=convo_data.get_group_energy_laplacian(i+1)
		group_energy_data[i][2]=atten_data.get_group_energy_laplacian(i+1)

	#df_eigen = pd.DataFrame(group_energy_data)

	print(group_energy_data)
	print(compelition)
	
	visializer = RoleGraph(all_data,3, compelition)
	visializer.plot_roles()

	plotter = ScatterMetricVsAccuracy(group_energy_data, compelition)
	plotter.regression_plot()

if __name__ == "__main__":
	main_graph_tool()
