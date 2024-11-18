from scipy.sparse.csgraph import laplacian
import sys
import os

# Get the parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Add the parent directory to sys.path
sys.path.insert(0, parent_dir)
from DataSet import DataSet
from main import load_csv, clean_compeletion_csv

import numpy as np

# Load data
data_folder_path = parent_dir + "/Data/"

def get_normed_l_mat(group_matrix:np.ndarray, verbose=False)->np.ndarray:
    # Get degree matrix 
    degree_matrix = np.diag(group_matrix.sum(axis=1)) 

    # Get Laplacian Matrix
    laplacian_matrix = degree_matrix - group_matrix

    # Get normalized Laplacian Matrix
    D_sqrt_inv = np.diag(1.0 / np.sqrt(degree_matrix.diagonal()))
    normalized_laplacian_matrix = D_sqrt_inv @ laplacian_matrix @ D_sqrt_inv

    # Scipy has "laplacian" function for this process
    scipy_normed_l_matrix = laplacian(group_matrix, normed=True)

    if verbose:
        print(f"degree_matrix: \n{degree_matrix}\n")
        print(f"laplacian_matrix: \n{laplacian_matrix}\n")
        print(f"normalized_laplacian_matrix: \n{normalized_laplacian_matrix}\n")
        print(f"scipy_normed_l_matrix: \n{scipy_normed_l_matrix}\n")

    return scipy_normed_l_matrix

def get_max_eigval_from_normed_l_mat(normed_l_mat:np.ndarray, verbose=False)->float:
    # find eigenvalues and eigenvectors of normalized Laplacian Matrix
    # one eigenvalues must be 0 cuz this is connected graph
    eig_vals, eig_vecs = np.linalg.eig(normed_l_mat)
    
    # find max eigenvalue, higher max eigen value means imbalanace
    max_eig_index = np.argmax(eig_vals)

    if verbose:
        print(f"eig_vals: \n{eig_vals}")
        print(f"max eig_vals and argmax: \n{eig_vals[max_eig_index]}, {max_eig_index}\n")
        print(f"eig_vecs: \n{eig_vecs}")
        print(f"max eig_vecs: \n{eig_vecs[max_eig_index]}\n")

    return eig_vals[max_eig_index]

def eigen_examples_example(verbose=False):

    # # example 1
    # sample_matrix = np.array([
    #     [0., 1., 1., 1.],
    #     [1., 0., 100., 100.],
    #     [1., 100., 0., 100.],
    #     [1., 100., 100., 0.]
    # ])

    # example 2
    sample_matrix = np.array([
        [0., 1., 1., 1.],
        [1., 0., 2., 2.],
        [1., 2., 0., 2.],
        [1., 2., 2., 0.]
    ])

    # # example 3
    # sample_matrix = np.array([
    #     [0., 1., 1., 1.],
    #     [1., 0., 1., 1.],
    #     [1., 1., 0., 1.],
    #     [1., 1., 1., 0.]
    # ])

    normalized_laplacian_matrix = get_normed_l_mat(
        group_matrix=sample_matrix,
        verbose=verbose
    )

    max_eig_val = get_max_eigval_from_normed_l_mat(
        normalized_laplacian_matrix,
        verbose=verbose
    )
    if verbose:
        print(f"sample_matrix: \n{sample_matrix}\n")
        print(f"max_eig_val: {max_eig_val}")

def max_eigen_dataset_example(verbose=False):

    prox_data = DataSet("proximity_graphs.json", directory=data_folder_path)
    convo_data = DataSet("conversation_graphs_directed.json", directory=data_folder_path)
    atten_data = DataSet("shared_attention_graphs.json", directory=data_folder_path)

    group_num = 11
    dataset_num = 3
    
    max_eig_val_mat = np.zeros((group_num, dataset_num))
    max_eig_val_sum_mat = np.zeros((group_num, 1))

    for i in range(group_num):
        prox_max_eig_val = max(prox_data.get_group_laplacian_eigenvalue_normed(i))
        convo_max_eig_val = max(convo_data.get_group_laplacian_eigenvalue_normed(i))
        atten_max_eig_val = max(atten_data.get_group_laplacian_eigenvalue_normed(i))

        max_eig_val_mat[i][0] = prox_max_eig_val
        max_eig_val_mat[i][1] = convo_max_eig_val
        max_eig_val_mat[i][2] = atten_max_eig_val

        max_eig_val_sum_mat[i] = sum(max_eig_val_mat[i])

    if verbose:  
        print(f"max_eig_val_mat: \n{max_eig_val_mat}\n")
        print(f"max_eig_val_sum_mat: \n{max_eig_val_sum_mat}\n")

    return max_eig_val_mat, max_eig_val_sum_mat

def get_sorted_array_and_ranknig(array:np.ndarray, is_descending=False):

    if is_descending:
        sort_multiplier = -1.0
    else: 
        sort_multiplier = +1.0

    # Compute rankings
    # ref : https://github.com/numpy/numpy/issues/8757#issuecomment-355126992
    rankings = np.argsort(np.argsort(sort_multiplier * array)) + 1 # Sort in descending order and make rankings 1-based
    # print(rankings)

    # Combine row sums and rankings for better visualization
    result = list(zip(array, rankings))

    return result

def evaluate_by_compelition_example(verbose=True):

    _, max_eig_val_sum_mat = max_eigen_dataset_example(verbose=False)

    # Flatten the array for ranking
    eig_val_ranking = get_sorted_array_and_ranknig(
        max_eig_val_sum_mat.flatten(),
        is_descending=False
    )

    # Load completion_time_and_accuracy csv data and Ranking again
    compelition = clean_compeletion_csv(
        load_csv(
            "completion_time_and_accuracy.csv",
             directory=data_folder_path
        )
    )
    compelition = np.array(compelition).reshape(11, 2)
    time_completion = compelition[:,0]
    acc_completion = compelition[:,1]

    time_ranking = get_sorted_array_and_ranknig(
        time_completion,
        is_descending=False
    )

    acc_ranking = get_sorted_array_and_ranknig(
        acc_completion,
        is_descending=True
    )

    # Print results
    if verbose:
        print("Ranking by Sum of EigenValues (Ascending Order):")
        for i, (value, rank) in enumerate(eig_val_ranking):
            print(f"Group: {i+1}\t| Value: {value:.8f}, Rank: {rank}")

        print("Ranking by Time (Ascending Order):")
        for i, (row_sum, rank) in enumerate(time_ranking):
            print(f"Group: {i+1}\t| Time: {row_sum:.8f}, Rank = {rank}")

        print("Ranking by Accuracy (Descending Order):")
        for i, (row_sum, rank) in enumerate(acc_ranking):
            print(f"Group: {i+1}\t| Acc: {row_sum:.8f}, Rank = {rank}")


if __name__ == "__main__":
    eigen_examples_example(verbose=False)
    max_eigen_dataset_example(verbose=False)
    evaluate_by_compelition_example(verbose=True)

    # # output
    # Ranking by Sum of EigenValues (Ascending Order):
    # Group: 1	| Value: 4.83873598, Rank: 9
    # Group: 2	| Value: 4.57345220, Rank: 5
    # Group: 3	| Value: 4.83544313, Rank: 8
    # Group: 4	| Value: 4.60684768, Rank: 7
    # Group: 5	| Value: 4.34280804, Rank: 1
    # Group: 6	| Value: 4.94899120, Rank: 11
    # Group: 7	| Value: 4.88952923, Rank: 10
    # Group: 8	| Value: 4.58296365, Rank: 6
    # Group: 9	| Value: 4.46735855, Rank: 4
    # Group: 10	| Value: 4.37164509, Rank: 2
    # Group: 11	| Value: 4.44153795, Rank: 3
    # Ranking by Time (Ascending Order):
    # Group: 1	| Time: 415.53900000, Rank = 2
    # Group: 2	| Time: 620.58600000, Rank = 7
    # Group: 3	| Time: 676.77800000, Rank = 10
    # Group: 4	| Time: 513.08500000, Rank = 4
    # Group: 5	| Time: 209.26500000, Rank = 1
    # Group: 6	| Time: 622.30800000, Rank = 8
    # Group: 7	| Time: 562.77300000, Rank = 6
    # Group: 8	| Time: 994.05400000, Rank = 11
    # Group: 9	| Time: 430.89100000, Rank = 3
    # Group: 10	| Time: 652.17100000, Rank = 9
    # Group: 11	| Time: 533.99600000, Rank = 5
    # Ranking by Accuracy (Descending Order):
    # Group: 1	| Acc: 67.85714286, Rank = 1
    # Group: 2	| Acc: 57.14285714, Rank = 4
    # Group: 3	| Acc: 53.57142857, Rank = 5
    # Group: 4	| Acc: 42.85714286, Rank = 10
    # Group: 5	| Acc: 50.00000000, Rank = 6
    # Group: 6	| Acc: 60.71428571, Rank = 2
    # Group: 7	| Acc: 44.44444444, Rank = 8
    # Group: 8	| Acc: 17.85714286, Rank = 11
    # Group: 9	| Acc: 60.71428571, Rank = 3
    # Group: 10	| Acc: 50.00000000, Rank = 7
    # Group: 11	| Acc: 42.85714286, Rank = 9
