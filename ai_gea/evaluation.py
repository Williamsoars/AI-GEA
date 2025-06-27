import numpy as np
import networkx as nx
from sklearn.metrics import f1_score
from typing import Dict, Callable, List
import statistics

# --- Funções que você enviou para deformação ---

def calcular_distancia_entre_nos(embedding_dict: Dict, label1: str, label2: str) -> float:
    vetor1 = np.array(embedding_dict[label1])
    vetor2 = np.array(embedding_dict[label2])
    return np.linalg.norm(vetor1 - vetor2)

def matrix_edge(G: nx.Graph) -> np.ndarray:
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

def calcular_deformacao(matrix: np.ndarray, embedding: Dict, G: nx.Graph) -> float:
    def encontrar_par_mais_proximo(matrix: np.ndarray, G: nx.Graph) -> tuple:
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

# --- Novas métricas ---

def reconstruction_error(G: nx.Graph, embedding: Dict) -> float:
    """
    Erro médio absoluto na reconstrução das arestas a partir do embedding.
    Calcula a diferença entre distância no embedding e presença/ausência de aresta.
    """
    nodes = sorted(G.nodes())
    erro = 0
    count = 0
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            dist_emb = np.linalg.norm(np.array(embedding[nodes[i]]) - np.array(embedding[nodes[j]]))
            tem_aresta = G.has_edge(nodes[i], nodes[j])
            erro += abs(dist_emb - (0 if tem_aresta else 1))
            count += 1
    return erro / count if count > 0 else float('inf')

def calcular_stress(G: nx.Graph, embedding: Dict) -> float:
    """
    Stress = soma das diferenças ao quadrado entre distâncias no grafo e no embedding.
    """
    dist_grafo = matrix_edge(G)
    nodes = sorted(G.nodes())
    stress_sum = 0.0
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            dist_emb = np.linalg.norm(np.array(embedding[nodes[i]]) - np.array(embedding[nodes[j]]))
            diff = dist_grafo[i][j] - dist_emb
            stress_sum += diff * diff
    return stress_sum

def calcular_f1_macro(G: nx.Graph, embedding: Dict, labels_true: Dict) -> float:
    """
    Exemplo simples para calcular F1 macro: prevê classes de nós via k-NN no embedding
    e compara com labels verdadeiros.
    
    labels_true: dict {node: label} fornecido pelo usuário.
    """
    from sklearn.neighbors import KNeighborsClassifier

    nodes = sorted(G.nodes())
    X = np.array([embedding[n] for n in nodes])
    y_true = np.array([labels_true[n] for n in nodes])

    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X, y_true)
    y_pred = knn.predict(X)

    return f1_score(y_true, y_pred, average='macro')

# --- Função principal para avaliação dos métodos ---

def avaliar_metodos(
    G: nx.Graph,
    metodos: List[str],
    embeddings_funcs: Dict[str, Callable[[nx.Graph], Dict]],
    labels_true: Dict = None,
    n_execucoes: int = 5
) -> Dict[str, Dict[str, float]]:
    """
    Avalia os métodos de embedding no grafo G executando n_execucoes vezes e calculando as métricas.
    
    Args:
        G: grafo NetworkX
        metodos: lista com nomes dos métodos (chaves de embeddings_funcs)
        embeddings_funcs: dict {nome_metodo: funcao(grafo) -> dict {node: vetor}}
        labels_true: rótulos verdadeiros para calcular F1 (opcional)
        n_execucoes: número de repetições para média
    
    Retorna:
        dict {metodo: {metricas}}
    """
    resultados = {}
    dist_grafo = matrix_edge(G)

    for metodo in metodos:
        f1s, stresses, recon_errors, deformacoes = [], [], [], []

        for _ in range(n_execucoes):
            embedding = embeddings_funcs[metodo](G)

            f1 = calcular_f1_macro(G, embedding, labels_true) if labels_true else 0.0
            stress = calcular_stress(G, embedding)
            recon_err = reconstruction_error(G, embedding)
            deform = calcular_deformacao(dist_grafo, embedding, G)

            f1s.append(f1)
            stresses.append(stress)
            recon_errors.append(recon_err)
            deformacoes.append(deform)

        resultados[metodo] = {
            "f1_macro": float(np.mean(f1s)),
            "stress": float(np.mean(stresses)),
            "reconstruction_error": float(np.mean(recon_errors)),
            "deformacao": float(np.mean(deformacoes))
        }

    return resultados

