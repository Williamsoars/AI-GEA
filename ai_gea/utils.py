import numpy as np
import networkx as nx
import statistics
from typing import Dict, Union

def calcular_distancia_entre_nos(embedding_dict: Dict, label1: str, label2: str) -> float:
    """Calcula a distância euclidiana entre dois nós no espaço de embedding.
    
    Args:
        embedding_dict: Dicionário de embeddings {nó: vetor}
        label1: ID do primeiro nó
        label2: ID do segundo nó
        
    Returns:
        Distância euclidiana entre os nós
    """
    vetor1 = np.array(embedding_dict[label1])
    vetor2 = np.array(embedding_dict[label2])
    return np.linalg.norm(vetor1 - vetor2)

def matrix_edge(G: nx.Graph) -> np.ndarray:
    """Calcula a matriz de distâncias entre todos os pares de nós.
    
    Args:
        G: Grafo de entrada
        
    Returns:
        Matriz de distâncias entre nós
    """
    spl = dict(nx.all_pairs_shortest_path_length(G))
    nodes = sorted(G.nodes())
    size = len(nodes)
    matrix = np.zeros((size, size))
    
    for i, node1 in enumerate(nodes):
        for j, node2 in enumerate(nodes):
            if node1 in spl and node2 in spl[node1]:
                matrix[i][j] = spl[node1][node2] if spl[node1][node2] != 0 else 0
            elif i != j:
                matrix[i][j] = size + 1  # Valor grande para nós desconectados
                
    return np.round(matrix, 3)

def calcular_deformacao(matrix: np.ndarray, embedding: Union[np.ndarray, Dict], G: nx.Graph) -> float:
    """Calcula a deformação entre o espaço original e o embedding.
    
    Args:
        matrix: Matriz de distâncias originais
        embedding: Vetor ou dicionário de embeddings
        G: Grafo de entrada
        
    Returns:
        Valor percentual da deformação média
    """
    def encontrar_par_mais_proximo(matrix: np.ndarray, G: nx.Graph) -> tuple:
        """Encontra o par de nós mais próximo no grafo original."""
        nodes = sorted(G.nodes())
        min_val = float('inf')
        min_pair = (None, None)
        
        for i in range(len(nodes)):
            for j in range(len(nodes)):
                if i != j and 0 < matrix[i][j] < min_val:
                    min_val = matrix[i][j]
                    min_pair = (nodes[i], nodes[j])
        return min_pair

    node1, node2 = encontrar_par_mais_proximo(matrix, G)
    if node1 is None or node2 is None:
        return None

    # Converte embedding para dicionário se necessário
    if isinstance(embedding, np.ndarray):
        embedding = {node: embedding[i] for i, node in enumerate(G.nodes())}

    dist = calcular_distancia_entre_nos(embedding, node1, node2)
    nodes = sorted(G.nodes())
    vec_deform = []
    soma_distancias_embedding = 0
    
    # Calcula soma total das distâncias no embedding
    for i in range(len(nodes)):
        for j in range(len(nodes)):
            if i != j:
                distq = calcular_distancia_entre_nos(embedding, nodes[i], nodes[j])
                soma_distancias_embedding += distq

    dist_N = dist / soma_distancias_embedding
    matrix_ideal = np.multiply(matrix, dist_N)

    # Calcula deformação para cada par de nós
    for i in range(len(nodes)):
        for j in range(len(nodes)):
            if i == j:
                continue
                
            distq = calcular_distancia_entre_nos(embedding, nodes[i], nodes[j])
            if matrix_ideal[i][j] != 0:
                deform = abs(matrix_ideal[i][j] - distq / soma_distancias_embedding)
            else:
                deform = abs(len(nodes) * dist - distq / soma_distancias_embedding)
            vec_deform.append(deform)
            
    return statistics.mean(vec_deform) * 100
