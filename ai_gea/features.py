import numpy as np
import networkx as nx
from recommender import 
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
import time
import tracemalloc

def benchmark_methods(G, methods_dict):
    """
    Mede o tempo de execução e uso de memória de cada método de embedding aplicado ao grafo G.

    Parâmetros:
    - G: networkx.Graph
    - methods_dict: dict com pares {nome: funcao_de_embedding(G)}

    Retorna:
    - dict com tempo (segundos) e memória (KB) por método
    """
    resultados = {}

    for nome, funcao in methods_dict.items():
        try:
            # Início da medição de memória
            tracemalloc.start()

            t0 = time.time()
            _ = funcao(G)  # Executa o método
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
