import os
import json
import argparse
import re
import numpy as np
from scipy.sparse.csgraph import laplacian
from pprint import pprint 


#define softmax with np
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis = 0)


class DataSet:
    "created a data set from a json"
    def __init__(self, file_name:str, directory:str="Data/", my_directed:bool=False):
        """
        Contructs from a string of the name of a json file. Can add a directory but the default is Data/
        
        :param file_name: Filename
        :param directory: Directory where file is stored.
        """
        self._name = file_name
        self._directory=directory
        """directory where file is stored"""
        self.json = None
        """json from the filename and directory"""
        self.list_adj_matrix = None
        """a list of numpy arrays that are adjacency lists"""
        self.load()
        self._json_to_adj_list()
        self.directed=my_directed


    def _extract_group_number(self, graph: str)->int:
        """Accepts a graph from JSON and determines by the group name as a number extracted from the "id" field - Credit: Diana
            :param: graph_id
            :returns: group number"""
        match = re.search(r'group-(\d+)', graph)
        return int(match.group(1)) if match else float('inf')  # If no group number, place it at the end

    def _load_json(self, file_name:str)->dict:
        """Load one json from Data directory
            :param file_name: Filename
            :returns:
                    dict: Of one item, the JSON Data as a dictionary"""
        if os.path.isfile(self._directory+file_name) and os.access(self._directory+file_name, os.R_OK):
            file_path=self._directory+file_name
            with open(file_path, 'r') as f:
                data = json.load(f)
                return data
        else:
            raise ValueError("JSON not found!") 

    def _sort_json(self):
        """Sort JSON data in the class and updates self.json. - Credit: Diana"""
        self.json["graphs"]=sorted(self.json["graphs"], key=lambda x: self._extract_group_number(x["id"]))

    def _create_np_matrix(self, graph:list)->np.ndarray:
        """Takes 1 group as a graph from the JSON dict then this creates numpy 4x4 ajacency matrix from the parse data. Index corresponds to nodes: A=0, B=1, C=2, D=3. Values are edges between the nodes. Accounts for directed and undirected graphs.
           :param graph: One graph from the JSON dictionary.
           :returns: np.ndarray - Adjaceny matrix in the for of a numpy array."""
                
        matrix = np.zeros((4, 4)) #intializes numpy array

        # creates matrix representation
        is_group_directed=graph["directed"]
        for edges in graph["edges"]:
            row=(ord(edges["source"])-65)
            col=(ord(edges["target"])-65)
            metadata=edges["metadata"]
            matrix[row][col]=metadata["weight"]
            if not is_group_directed: #If graph is not directed, populates the lower left triangle.
                matrix[col][row]=metadata["weight"]
        return matrix
    
    def _json_to_adj_list(self):
        """Converts JSON dictionary loaded from the contructor turns it into a list of numpy darrays. Stores the darray in the DataSet object as self.list_adj_matrix."""
        list=[]
        
        for i in range(len(self.json["graphs"])):
            matrix=self._create_np_matrix(self.json["graphs"][i])
            list.append(matrix)
        self.list_adj_matrix=list

    def set_directory(self, new_directoy:str, new_file_name:str=""):
        """Update the DataSet object by change the folder to another folder or folder path relative to the current directory. Has the option of changing filename as well. Reloads data by calling load() and _list_adj_matrix()
            :param new_directory: New Directory Name
            :param new_file_name: New File Name (optional)
            """
        self._directory=new_directoy
        if not new_directoy=="":
            self._name=new_file_name
        # else it will stay the same if there is no new file name
        self.load()
        self.list_adj_matrix()

    def set_new_file(self, new_file_name):
        """Updates the DataSet object by changing the file name. Reloads data in self.json and self.list_adj_matrix.
            :param new_file_name: New File Name"""
        self.json=self.load_sorted_json(new_file_name)
        self.load()
        self.list_adj_matrix()

    def load(self):
        """Load JSON from the given filename is the DataSet object. Sort the JSON by groupname and stores it in the DataSet object under self.json."""
        self.json=self._load_json(self._name)
        self._sort_json()

    # Thinking about deleting this 
    def load_from_arg(self):
        """Loads JSON file from first argument in args. Overrides old JSON file in DataSet object self.json."""
        parser = argparse.ArgumentParser(description='Load from a JSON file.')
        parser.add_argument('json_file', type=str, help='Path to the JSON file containing graph data')
        args = parser.parse_args()

        # Load the JSON data from the specified file
        with open(args.json_file, 'r') as f:
            self.json = json.load(f)
 
    def print_json(self):
        """Prints JSON in a more readable way. Puts self.json in stdout (standard output)."""
        pprint(self.json)

    def print_adj_matrix(self):
        """Prints list of numpy darray in a \'more\' readable way. Puts self.list_adj_matrix in stdout (standard output)."""
        pprint(self.list_adj_matrix)

    def get_sum_group_nodes(self, group_num:int)->list:
        """Calculate the total of the node value for a group. Does this by summing row of adjaceny array for each row.
            :param group_num: Group number
            :returns: A list of node values for a single group that is ordered by the node name alphabetically."""
        matrix=self.list_adj_matrix[group_num-1]
        nodes=[]
        for V in matrix:
            sum=0
            for edge in V:
                sum+=edge
            nodes.append(sum)
        return nodes
    
    def get_sum_all_nodes(self)->list:
        """Creates a list of all participants data ordered by group (and within the group the participants is order alphabetically).
        0:group 1, particpant a, group 1, participant b, group 1, participant c, group 1, participant d, group 2, participant a...
        :return: Ordered list of all nodes."""
        all_nodes=[]

        for i in range(len(self.json["graphs"])):
            matrix=self._create_np_matrix(self.json["graphs"][i])
            for v in matrix:
                sum=0
                for edge in v:
                    sum+=edge
                all_nodes.append(sum)
        return all_nodes

    def get_group_matrix(self, group_num:int)->np.ndarray:
        """
        Takes in group number and returns adjaceny matrix
        
        :returns: Adjacency list of the group as a numpy array.
        """
        return self.list_adj_matrix[group_num-1]
    
    # Maybe the data set is doing to much?
    def get_group_eigenvalue(self, group_num:int)->np.ndarray:
        """Eigenvalue from a group number.

        #### Note: Not sure it's so useful for conversation graph? Aren't the rows the same number. Maybe I'm wrong. Talk to me about it.
        :param group_num: Group Number
        :returns: Eigenvalues as a numpy array."""
        matrix=self.get_group_matrix(group_num)
        return np.linalg.eig(matrix).eigenvalues

    def get_group_eigenvalue_l2norm(self, group_num:int)->np.ndarray:
        matrix=self._normalize_matrix(self.get_group_matrix(group_num),2)
        return np.linalg.eig(matrix).eigenvalues

    def get_group_energy(self, group_num:int):
        #initialize energy
        energy = 0

        #get the group eigen values with the input as a np array
        eigen_vector = self.get_group_eigenvalue(group_num)
        
        #sum of absolute values of eigen values
        for eigen in eigen_vector:
            energy += abs(eigen)

        return energy
    
   
    def get_group_laplacian_eigenvalue(self, group_num:int)->np.ndarray:
        matrix=laplacian(self.get_group_matrix(group_num))
        return np.linalg.eig(matrix).eigenvalues

    def get_group_laplacian_eigenvalue_normed(self, group_num:int)->np.ndarray:
        if self.directed:
            matrix=laplacian(self.get_group_matrix(group_num), normed=True)
        else:
            print(self._name)
            matrix=laplacian(self.get_group_matrix(group_num), normed=True, symmetrized=True)
        return np.linalg.eig(matrix).eigenvalues

    def get_group_laplacian_eigenvalue_normed_l2(self, group_num:int)->np.ndarray:
        matrix=laplacian(self._normalize_matrix(self.get_group_matrix(group_num),2))
        return np.linalg.eig(matrix).eigenvalues

    def get_group_laplacian_energy(self, group_num: int):
        # initialize energy
        energy = 0

        # get the group eigenvalues with the input as a np array
        eigen_vector = self.get_group_laplacian_eigenvalue_normed(group_num)

        #fix this
        if self.directed:   
            group_adj_matrix = laplacian(self.get_group_matrix(group_num), normed=True)
        else:
            group_adj_matrix = laplacian(self.get_group_matrix(group_num), normed=True, symmetrized=True)
        ##comeback here

        in_degrees = np.sum(group_adj_matrix, axis=0)  # Sum along columns
        out_degrees = np.sum(group_adj_matrix, axis=1)  # Sum along rows

        #total degree per node
        total_degrees = in_degrees + out_degrees

        num_nodes = group_adj_matrix.shape[0]

        average_degree = np.sum(total_degrees) / num_nodes

        # return energy
        # return np.sum(np.pow((eigen_vector - average_degree), 2))
        return np.sum(abs(eigen_vector - average_degree))

        # def get_group_energy_laplacian(self, group_num:int):
        #     #initialize energy
        #     energy = 0

        #     #get the group eigen values with the input as a np array
        #     # eigen_vector = self.get_group_laplacian_eigenvalue_normed_l2(group_num) #changed
        #     # eigen_vector = self.get_group_laplacian_eigenvalue_normed(group_num)
        #     eigen_vector = self.get_group_laplacian_eigenvalue(group_num)

        #     #sum of absolute values of eigen values
        #     for eigen in eigen_vector:
        #         energy += abs(eigen)

        #     return energy

    def get_group_energy_laplacian(self, group_num:int):
        #initialize energy
        energy = 0

        #get the group eigen values with the input as a np array
        eigen_vector = self.get_group_laplacian_eigenvalue_normed(group_num)

        lambda_mean = np.mean(eigen_vector)
        
        # for eigen in eigen_vector:
        #     energy += (lambda_i - eigen_vector[eigen])**2

        # return energy
        return np.sum(np.pow((eigen_vector - lambda_mean), 2))

    def _normalize_matrix(se1lf, group_matrix:np.ndarray, type:int)->np.ndarray:
        """Takes a group number L? Normalization of the adjacency list
        :param group_matrix: (ndarray) matrix of one group
        :param type: takes a number that indicates l1 or l2 normalization. 1=l1, 2=l2
        :returns: (str) matrix normalize with l2"""

        #this epislon is here to prevent divide by zero - smallest epislon for float64
        #if you want to change the percision to float32 replace float 64 with it.
        #compute l2 on each column
        epsilon = np.finfo(np.float64).eps
        if type == 2:
            l2_norms = np.maximum(group_matrix, epsilon)
            return group_matrix / (np.linalg.norm(l2_norms, axis=0))
        
        if type == 1:
            #### By Column - it gave me all 1s after the sum wtf
            l1_norms = np.sum(np.abs(group_matrix), axis=0)

            #prevent divide by zero
            l1_norms = np.maximum(l1_norms, epsilon)

            return group_matrix / l1_norms

            #### By Row - feels wrong, worth a shot
            # l1_norm_rows = np.sum(np.abs(group_matrix), axis=1)

            # return group_matrix / l1_norm_rows[:, np.newaxis]  # Use broadcasting for row-wise division

        
        #if you made it here you messed up
        return None
    
    def l1_normalize(x):
        return x / np.linalg.norm(x, ord=1)

    def get_sum_all_nodes_normalize(self, num:int):
        """
        Returns a 1-dimensional vector of sums of normalized edges for all nodes.
        """
        all_nodes = []

        # Iterate over each node's adjacency list
        if self.directed:
            for adj_matrix in self.list_adj_matrix:  # Check if 11 is the correct number of iterations
                vector=np.zeros(4)
                for i, row in enumerate(adj_matrix):
                    #sum all the rows
                    vector[i]=row.sum()/(len(adj_matrix)-1)

                # print(vector)
                # all_nodes.append(self.l1_normalize(vector))# Append the sum for the current node
                norm_vector=vector / np.linalg.norm(vector, ord=num)
                for i in np.nditer(norm_vector):
                    all_nodes.append(i)
        else:
            # print(f'sum all nodes normalized... {self._name} = {self.directed}')
            for adj_matrix in self.list_adj_matrix:  # Check if 11 is the correct number of iterations
                norm_adj = self._normalize_matrix(adj_matrix, num)  # Normalize adjacency matrix
                for row in norm_adj:
                    #sum all the rows
                    all_nodes.append(row.sum())# Append the sum for the current node
    
    
        
        # Return all nodes as a 1D list (flattened vector)
        return all_nodes

