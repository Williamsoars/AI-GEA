from typing import TYPE_CHECKING, Dict, Any
if TYPE_CHECKING:
    from .recommender import EmbeddingRecommender  # Só para type checking

def extrair_features_grafo(G: nx.Graph) -> Dict[str, Union[float, int]]:
    """Retorno tipado mais precisamente"""
    """
    Extract features from a graph for embedding recommendation.
    
    Args:
        G: NetworkX graph
        
    Returns:
        Dictionary of graph features
    """
    features = {
        "n_nodes": G.number_of_nodes(),
        "n_edges": G.number_of_edges(),
        "density": nx.density(G),
        "avg_clustering": nx.average_clustering(G),
        "avg_degree": np.mean([d for n, d in G.degree()]),
        "transitivity": nx.transitivity(G),
    }
    
    # Handle disconnected graphs
    if nx.is_connected(G):
        features["diameter"] = nx.diameter(G)
        features["avg_shortest_path"] = nx.average_shortest_path_length(G)
    else:
        # Calculate for connected components
        features["diameter"] = max([nx.diameter(G.subgraph(c)) 
                                  for c in nx.connected_components(G) if len(c) > 1])
        avg_paths = []
        for c in nx.connected_components(G):
            if len(c) > 1:
                subgraph = G.subgraph(c)
                avg_paths.append(nx.average_shortest_path_length(subgraph))
        features["avg_shortest_path"] = np.mean(avg_paths) if avg_paths else 0
    
    return features

def benchmark_methods(G: nx.Graph, methods_dict: Dict[str, Callable]) -> Dict[str, Dict[str, Any]]:
    """
    Measure execution time and memory usage of each embedding method applied to graph G.

    Parameters:
    - G: networkx.Graph
    - methods_dict: dict with {name: embedding_function(G)} pairs

    Returns:
    - dict with time (seconds) and memory (KB) per method
    """
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
