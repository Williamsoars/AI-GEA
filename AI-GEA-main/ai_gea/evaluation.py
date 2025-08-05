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
    # Converter embedding para dicionário se for numpy array
    if isinstance(embedding, np.ndarray):
        nodes = sorted(G.nodes())
        embedding = {nodes[i]: embedding[i] for i in range(len(nodes))}
    
    # Verificar tipos
    if not all(isinstance(k, (int, np.integer)) for k in embedding.keys()):
        embedding = {int(k): v for k, v in embedding.items()}
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
    # Verificação de tipos
    if not isinstance(embedding, dict):
        if isinstance(embedding, np.ndarray):
            embedding = {i: embedding[i] for i in range(len(embedding))}
        else:
            raise TypeError("Embedding deve ser dict ou numpy array")
    """Enhanced reconstruction error with edge weight consideration"""
    nodes = sorted(G.nodes())
    erro = 0
    count = 0
    
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            u, v = nodes[i], nodes[j]
            dist_emb = np.linalg.norm(np.array(embedding[u]) - np.array(embedding[v]))
            
            # Handle both weighted and unweighted graphs
            if G.has_edge(u, v):
                weight = G[u][v].get('weight', 1.0)
                erro += abs(dist_emb - weight)
            else:
                erro += abs(dist_emb - 1.0)  # Default distance for non-edges
            count += 1
            
    return erro / count if count > 0 else float('inf')

def calcular_stress(G: nx.Graph, embedding: Dict) -> float:
    """Stress metric with normalization"""
    dist_grafo = matrix_edge(G)
    nodes = sorted(G.nodes())
    stress_sum = 0.0
    count = 0
    
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            dist_emb = np.linalg.norm(np.array(embedding[nodes[i]]) - np.array(embedding[nodes[j]]))
            diff = dist_grafo[i][j] - dist_emb
            stress_sum += diff * diff
            count += 1
            
    return stress_sum / count if count > 0 else float('inf')

def calcular_coesao_clusters(embedding: Dict, n_clusters: int = 3) -> float:
    """Measure cluster cohesion in embedding space with robust type checking"""
    try:
        # Convert embedding to numpy array safely
        if isinstance(embedding, dict):
            X = np.array(list(embedding.values()))
        elif isinstance(embedding, np.ndarray):
            X = embedding
        else:
            return np.nan  # Invalid input type
            
        # Check for valid data
        if len(X) < n_clusters or np.isnan(X).any():
            return np.nan
            
        # Perform clustering
        kmeans = KMeans(n_clusters=min(n_clusters, len(X)))
        labels = kmeans.fit_predict(X)
        return silhouette_score(X, labels)
        
    except Exception as e:
        print(f"Error in calcular_coesao_clusters: {str(e)}")
        return np.nan

def calcular_f1_macro(G: nx.Graph, embedding: Dict, labels_true: Dict) -> float:
    """Enhanced node classification with cross-validation"""
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import cross_val_score
    
    nodes = sorted(G.nodes())
    X = np.array([embedding[n] for n in nodes])
    y_true = np.array([labels_true[n] for n in nodes])

    if len(np.unique(y_true)) < 2:
        return 0.0  # Need at least two classes
    
    knn = KNeighborsClassifier(n_neighbors=3)
    scores = cross_val_score(knn, X, y_true, cv=3, scoring='f1_macro')
    return np.mean(scores)

# --- Main evaluation function ---
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
    
    Returns:
        dict {metodo: {metricas}}
    """
    resultados = {}
    dist_grafo = matrix_edge(G)

    for metodo in metodos:
        f1s, stresses, recon_errors, deformacoes, coesoes = [], [], [], [], []

        for _ in range(n_execucoes):
            embedding = embeddings_funcs[metodo](G)

            # Calculate all metrics
            f1 = calcular_f1_macro(G, embedding, labels_true) if labels_true else 0.0
            stress = calcular_stress(G, embedding)
            recon_err = reconstruction_error(G, embedding)
            deform = calcular_deformacao(dist_grafo, embedding, G)
            coesao = calcular_coesao_clusters(embedding)

            f1s.append(f1)
            stresses.append(stress)
            recon_errors.append(recon_err)
            deformacoes.append(deform)
            coesoes.append(coesao)

        resultados[metodo] = {
            "f1_macro": float(np.mean(f1s)),
            "stress": float(np.mean(stresses)),
            "reconstruction_error": float(np.mean(recon_errors)),
            "deformacao": float(np.mean(deformacoes)),
            "coesao_clusters": float(np.mean(coesoes)),
            "norma": float(np.linalg.norm(np.array(list(embedding.values()))))
        }

    return resultados
