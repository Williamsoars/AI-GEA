import numpy as np
import networkx as nx
import statistics

def calcular_distancia_entre_nos(embedding_dict, label1, label2):
    vetor1 = np.array(embedding_dict[label1])
    vetor2 = np.array(embedding_dict[label2])
    return np.linalg.norm(vetor1 - vetor2)

def matrix_edge(G):
    spl = dict(nx.all_pairs_shortest_path_length(G))
    nodes = sorted(G.nodes())
    size = len(nodes)
    matrix = np.zeros((size, size))
    for i, node1 in enumerate(nodes):
        for j, node2 in enumerate(nodes):
            if node1 in spl and node2 in spl[node1]:
                matrix[i][j] = spl[node1][node2] if spl[node1][node2] != 0 else 0
            elif i != j:
                matrix[i][j] = size + 1
    return np.round(matrix, 3)

def calcular_deformacao(matrix, embedding, G):
    def matrix_checking(matrix, G):
        nodes = sorted(G.nodes())
        min_val = float('inf')
        min_pair = (None, None)
        for i in range(len(nodes)):
            for j in range(len(nodes)):
                if i != j and 0 < matrix[i][j] < min_val:
                    min_val = matrix[i][j]
                    min_pair = (nodes[i], nodes[j])
        return min_pair

    node1, node2 = matrix_checking(matrix, G)
    if node1 is None or node2 is None:
        return None

    if isinstance(embedding, np.ndarray):
        embedding = {node: embedding[i] for i, node in enumerate(G.nodes())}

    dist = calcular_distancia_entre_nos(embedding, node1, node2)
    nodes = sorted(G.nodes())
    vec_deform = []
    soma_distancias_embedding = 0
    for i in range(len(nodes)):
        for j in range(len(nodes)):
            if i != j:
                distq = calcular_distancia_entre_nos(embedding, nodes[i], nodes[j])
                soma_distancias_embedding += distq

    dist_N = dist / soma_distancias_embedding
    matrix_ideal = np.multiply(matrix, dist_N)

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
