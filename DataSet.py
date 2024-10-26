import os
import json
import argparse
import re
import numpy as np
from pprint import pprint

class DataSet:
    "created a data set from a json"
    def __init__(self, file_name:str, directory:str="Data/"):
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

    def _extract_group_number(self, graph: str)->int:
        """Accepts a graph from JSON and determines by the group name as a number extracted from the "id" field - Credit: Diana
            :param: graph_id
                Return: group number"""
        match = re.search(r'group-(\d+)', graph)
        return int(match.group(1)) if match else float('inf')  # If no group number, place it at the end

    def _load_json(self, file_name:str)->dict:
        """Load one json from Data directory
            :param file_name: Filename
                Returns:
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

    def _create_np_matrix(self, graph:list)->list:
        """Takes 1 group as a graph from the JSON dict then this creates numpy 4x4 ajacency matrix from the parse data. Index corresponds to nodes: A=0, B=1, C=2, D=3. Values are edges between the nodes. Accounts for directed and undirected graphs.
           :param graph: One graph from the JSON dictionary.
            Returns:
                np.darray: Adjaceny matrix in the for of a numpy array."""
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
                Returns: A list of node values for a single group that is ordered by the node name alphabetically."""
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
            Return: Ordered list of all nodes."""
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
        """Takes in group number and returns adjaceny matrix"""
        return self.list_adj_matrix[group_num-1]
