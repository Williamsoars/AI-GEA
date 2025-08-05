from typing import TYPE_CHECKING, Dict, Any
import networkx as nx
from typing import Union
import numpy as np
import scipy.stats
if TYPE_CHECKING:
    from .recommender import EmbeddingRecommender

def extrair_features_grafo(G: nx.Graph) -> Dict[str, Union[float, int]]:
    """
    Extract comprehensive features from a graph for embedding recommendation.
    
    Args:
        G: NetworkX graph
        
    Returns:
        Dictionary of graph features with additional metrics
    """
    degrees = [d for n, d in G.degree()]
    clustering_coeffs = list(nx.clustering(G).values())
    
    features = {
        # Basic features
        "n_nodes": G.number_of_nodes(),
        "n_edges": G.number_of_edges(),
        "density": nx.density(G),
        "avg_clustering": nx.average_clustering(G),
        "avg_degree": np.mean(degrees),
        "transitivity": nx.transitivity(G),
        
        # New centrality measures
        "degree_assortativity": nx.degree_assortativity_coefficient(G),
        "degree_entropy": float(scipy.stats.entropy(degrees)),
        
        # Path-based features
        "avg_eccentricity": np.mean(list(nx.eccentricity(G).values())) if nx.is_connected(G) else -1,
        
        # New ratio features
        "nodes_edges_ratio": G.number_of_nodes() / G.number_of_edges() if G.number_of_edges() > 0 else 0,
        "max_min_degree_ratio": max(degrees) / min(degrees) if min(degrees) > 0 else 0,
        
        # Clustering statistics
        "clustering_std": np.std(clustering_coeffs) if clustering_coeffs else 0,
        "clustering_entropy": float(scipy.stats.entropy(clustering_coeffs)) if clustering_coeffs else 0,
    }
    
    # Handle path-based features for disconnected graphs
    if nx.is_connected(G):
        features.update({
            "diameter": nx.diameter(G),
            "avg_shortest_path": nx.average_shortest_path_length(G),
            "radius": nx.radius(G)
        })
    else:
        # Calculate for largest connected component
        largest_cc = max(nx.connected_components(G), key=len)
        subgraph = G.subgraph(largest_cc)
        features.update({
            "diameter_lcc": nx.diameter(subgraph),
            "avg_shortest_path_lcc": nx.average_shortest_path_length(subgraph),
            "radius_lcc": nx.radius(subgraph)
        })
    
    return features

def benchmark_methods(G: nx.Graph, methods_dict: Dict[str, callable]) -> Dict[str, Dict[str, Any]]:
    """
    Measure execution time and memory usage of each embedding method applied to graph G.

    Parameters:
    - G: networkx.Graph
    - methods_dict: dict with {name: embedding_function(G)} pairs

    Returns:
    - dict with time (seconds) and memory (KB) per method
    """
    import time
    import tracemalloc
    resultados = {}

    for nome, funcao in methods_dict.items():
        try:
            # Start memory measurement
            tracemalloc.start()

            t0 = time.time()
            _ = funcao(G)  # Execute the method
            t1 = time.time()

            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            resultados[nome] = {
                "tempo_segundos": round(t1 - t0, 4),
                "memoria_kb": round(peak / 1024, 2)
            }

        except Exception as e:
            resultados[nome] = {
                "erro": str(e),
                "tempo_segundos": None,
                "memoria_kb": None
            }

    return resultados

