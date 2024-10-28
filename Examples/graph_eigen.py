# python eigenvalues of graph example

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

def create_weighted_graph():
    """Create a sample weighted undirected graph"""
    G = nx.Graph()
    edges_with_weights = [('A', 'B', 2), ('A', 'C', 3), 
                         ('B', 'C', 1), ('C', 'D', 4)]
    G.add_weighted_edges_from(edges_with_weights)
    return G

def plot_weighted_graph(G):
    """Plot the weighted graph with labels"""
    plt.figure(figsize=(8, 6))
    pos = nx.spring_layout(G)
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=500)
    
    # Draw edges
    nx.draw_networkx_edges(G, pos)
    
    # Draw labels
    nx.draw_networkx_labels(G, pos)
    edge_labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels)
    
    plt.title("Weighted Graph")
    plt.axis('off')

def calculate_eigenvalues(G):
    """Calculate and return eigenvalues of the graph"""
    adjacency_matrix = nx.adjacency_matrix(G, weight='weight').todense()
    return np.linalg.eigvals(adjacency_matrix)

def plot_eigenvalues(eigenvalues):
    """Plot the eigenvalues in complex plane"""
    plt.figure(figsize=(8, 6))
    plt.scatter(eigenvalues.real, eigenvalues.imag)
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    plt.grid(True)
    plt.title("Eigenvalues in Complex Plane")
    plt.xlabel("Real Part")
    plt.ylabel("Imaginary Part")

def main():
    # Create and analyze the graph
    G = create_weighted_graph()
    
    # Plot the graph
    plot_weighted_graph(G)
    
    # Calculate and plot eigenvalues
    eigenvalues = calculate_eigenvalues(G)
    print("Eigenvalues of the weighted graph:", eigenvalues)
    plot_eigenvalues(eigenvalues)
    
    plt.show()

if __name__ == "__main__":
    main()