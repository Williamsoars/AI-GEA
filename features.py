import numpy as np
import networkx as nx

def extrair_features_grafo(G):
    features = {
        "n_nodes": G.number_of_nodes(),
        "n_edges": G.number_of_edges(),
        "density": nx.density(G),
        "avg_clustering": nx.average_clustering(G),
        "avg_degree": np.mean([d for n, d in G.degree()]),
        "transitivity": nx.transitivity(G),
    }
    
    # Tratamento para grafos desconectados
    if nx.is_connected(G):
        features["diameter"] = nx.diameter(G)
        features["avg_shortest_path"] = nx.average_shortest_path_length(G)
    else:
        # Calcula para componentes conectados
        features["diameter"] = max([nx.diameter(c) for c in nx.connected_components(G) if len(c) > 1])
        avg_paths = []
        for c in nx.connected_components(G):
            if len(c) > 1:
                subgraph = G.subgraph(c)
                avg_paths.append(nx.average_shortest_path_length(subgraph))
        features["avg_shortest_path"] = np.mean(avg_paths) if avg_paths else 0
    
    return features
