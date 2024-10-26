import csv
from DataSet import DataSet #custom class

import numpy as np
from scipy.sparse import csgraph


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

# stubs
def map_label_participant():
    """## To Do?: make a mapping eventually
    ### {
        'person1group1':0,
        'person2group1':1,
        'person3group1':2,
        'person4group1':3,
        'person1group2':4,
    ### }
    ## Don't do it until we know we need it"""
    return None

def stub(my_string:str="stub")->str:
    """function stub
       :param my_string: a string
        Returns:
            str: a string"""
    return my_string

def main():
    num_groups=11
    total_participants=num_groups*4

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

    # How to use Dataset
    # print json
    # prox_data.print_json() # it's long

    # print list of adj matrix
    # prox_data.print_adj_matrix() # it's long
    
    # Gets numpy ajacency matrix from a group
    print(prox_data.get_group_matrix(2))

    # calculate nodes of group and puts it in a list - order of participant: a, b, c, d
    print(prox_data.get_sum_group_nodes(2))

    # calculate all nodes of dataset in a list - order by group number (and participant is ordered is a, b, c, d)
    print(prox_data.get_sum_all_nodes())

    #numpy array where data set is column and nodes are rows
    all_data=np.zeros((total_participants,3))
    for i in range(total_participants):
        all_data[i][0]=prox_data.get_sum_all_nodes()[i]
        all_data[i][1]=convo_data.get_sum_all_nodes()[i]
        all_data[i][2]=convo_data.get_sum_all_nodes()[i]

    #numpy array where nodes are rows and data set is column - probably we don't need this?
    # all_data2=np.zeros((3,num_groups*4))
    # for i in range(num_groups*4):
    #     all_data2[0][i]=prox_data.get_sum_all_nodes()[i]
    #     all_data2[1][i]=convo_data.get_sum_all_nodes()[i]
    #     all_data2[2][i]=convo_data.get_sum_all_nodes()[i]

    print(all_data)

    # To Think About: get some sort of average for conversation/proximity/shared attention divided by time from completion data (maybe I'm wrong)
    # To group members: to try stuff out on your own, maybe copy main.py as a templete since it loaded and transformt data
    #  rename your file so there aren't merge conflicts. Also, we should really talk to each other to figure out how to use kmeans


if __name__ == "__main__":
    main()