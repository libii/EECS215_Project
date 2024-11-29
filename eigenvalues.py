import csv
from DataSet import DataSet #custom class

import numpy as np
from pprint import pprint 

def load_csv(file_name:str, directory:str="Data/")->list:
    """Load CSV from Data directory.
    :param file_name: Filename
    :param directory: Directory where file is stored.
        Return:
            CSV data as a list contain a list for rows. Each row represents a group."""
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
            Returns:
                List of tuples (time (in secs):float, accuracy as percentage:float) with each row represents a group."""
    clean_data=[]
    data=data[1:]
    for row in data:
        junk, junk2, time = row[1].split()
        hours, minutes, seconds = map(float, time.split(':'))
        in_seconds = minutes * 60 + seconds
        clean_data.append((in_seconds,float(row[2])))
    return clean_data

def main():
    num_groups=11
    data_sets=3

    #loads csv data
    compelition=clean_compeletion_csv(load_csv("completion_time_and_accuracy.csv"))

    # print time of first group - note: first in the tuple
    print(compelition[0][0])

    # print accuracy of first group - note: seconds in the tuple
    print(compelition[0][1])

    #load json data - must give a file name, can also take another folder relative to the location of the current file that calls it in the directory
    convo_data=DataSet("conversation_graphs.json")
    prox_data=DataSet("proximity_graphs.json")
    atten_data=DataSet("shared_attention_graphs.json")

    #UNNORMALIZED
    # group_eigan =np.zeros((num_groups, data_sets))
    group_eigan = []
    for i in range(num_groups):
        row=[]
        row.append(np.around(prox_data.get_group_eigenvalue(i+1), decimals=1))
        row.append(np.around(convo_data.get_group_eigenvalue(i+1), decimals=1))
        row.append(np.around(atten_data.get_group_eigenvalue(i+1), decimals=1))
        group_eigan.append(row)
    print(prox_data.get_group_eigenvalue(i+1))
    print("\nRaw / Unnormalized")
    print(29 * "_")
    pprint(group_eigan)

    # Normalize
    normalized_group_eigan=[]
    for i in range(num_groups):
        row=[]
        row.append(np.around(prox_data.get_group_eigenvalue_l2norm(i+1), decimals=1))
        row.append(np.around(convo_data.get_group_eigenvalue_l2norm(i+1), decimals=1))
        row.append(np.around(atten_data.get_group_laplacian_eigenvalue(i+1), decimals=1))
        normalized_group_eigan.append(row)
    print("\nNormalized")
    print(29 * "_")
    pprint(normalized_group_eigan)

    # Laplacian
    normalized_group_eigan=[]
    for i in range(num_groups):
        row=[]
        row.append(np.around(prox_data.get_group_laplacian_eigenvalue(i+1), decimals=1))
        row.append(np.around(convo_data.get_group_laplacian_eigenvalue(i+1), decimals=1))
        row.append(np.around(atten_data.get_group_laplacian_eigenvalue(i+1), decimals=1))
        normalized_group_eigan.append(row)
    print("\nLaplacian")
    print(29 * "_")
    pprint(normalized_group_eigan)
    








if __name__ == "__main__":
    main()