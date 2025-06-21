import numpy as np
import networkx as nx

def extrair_features_grafo(G):
    return {
        "n_nodes": G.number_of_nodes(),
        "n_edges": G.number_of_edges(),
        "density": nx.density(G),
        "avg_clustering": nx.average_clustering(G),
        "avg_degree": np.mean([d for n, d in G.degree()]),
        "diameter": nx.diameter(G) if nx.is_connected(G) else 0,
        "avg_shortest_path": nx.average_shortest_path_length(G) if nx.is_connected(G) else 0,
        "transitivity": nx.transitivity(G),
    }
