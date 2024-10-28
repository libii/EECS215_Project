# python undirected graph example

import networkx as nx
import matplotlib.pyplot as plt

def generate_sample_graph():
    """Create a sample undirected graph"""
    G = nx.Graph()
    
    # Add nodes
    nodes = ['A', 'B', 'C', 'D']
    G.add_nodes_from(nodes)
    
    # Add edges with weights
    edges = [('A', 'B', 0.8), ('B', 'C', 0.6), 
            ('C', 'D', 0.9), ('D', 'A', 0.7),
            ('B', 'D', 0.5)]
    G.add_weighted_edges_from(edges)
    
    return G

def plot_graph(G):
    """Plot the undirected graph with weights"""
    # Set up the plot
    plt.figure(figsize=(8, 6))
    
    # Create a layout for the nodes
    pos = nx.spring_layout(G)
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', 
                          node_size=500)
    
    # Draw edges
    nx.draw_networkx_edges(G, pos)
    
    # Draw node labels
    nx.draw_networkx_labels(G, pos)
    
    # Draw edge labels
    edge_labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels)
    
    plt.title("Sample Undirected Graph")
    plt.axis('off')

def main():
    # Generate and plot sample graph
    G = generate_sample_graph()
    plot_graph(G)
    plt.show()

if __name__ == "__main__":
    main()
