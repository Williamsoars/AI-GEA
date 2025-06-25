import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import numpy as np

def plot_embedding(embedding, labels=None, method='TSNE', title='Embedding Visualization'):
    """
    Visualiza um embedding em 2D com redução de dimensionalidade.

    Parâmetros:
    - embedding: np.ndarray (n_nodes, n_features)
    - labels: lista ou array opcional com rótulos dos nós
    - method: 'TSNE' ou 'PCA'
    - title: título do gráfico
    """
    if method == 'TSNE':
        reducer = TSNE(n_components=2, random_state=42, perplexity=30)
    elif method == 'PCA':
        reducer = PCA(n_components=2)
    else:
        raise ValueError("Método inválido. Use 'TSNE' ou 'PCA'.")

    reduced = reducer.fit_transform(embedding)

    plt.figure(figsize=(8, 6))
    if labels is not None:
        labels = np.array(labels)
        for label in np.unique(labels):
            idx = labels == label
            plt.scatter(reduced[idx, 0], reduced[idx, 1], label=f"Label {label}", s=30)
        plt.legend()
    else:
        plt.scatter(reduced[:, 0], reduced[:, 1], s=30)

    plt.title(f"{title} ({method})")
    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    
def plot_comparacao_scores(dict_scores, titulo='Comparação de Scores por Método'):
    """
    Plota um gráfico de barras comparando scores de diferentes métodos.

    Parâmetros:
    - dict_scores: dicionário com formato:
        {
            'Node2Vec': {'f1_macro': 0.87, 'deformacao': 0.45},
            'DeepWalk': {'f1_macro': 0.85, 'deformacao': 0.41},
            ...
        }
    - titulo: Título do gráfico
    """
    if not dict_scores:
        print("Nenhum dado fornecido para plotagem.")
        return

    # Coletar todas as métricas únicas
    metricas = set()
    for valores in dict_scores.values():
        metricas.update(valores.keys())
    metricas = sorted(metricas)

    metodos = list(dict_scores.keys())
    n_metricas = len(metricas)
    n_metodos = len(metodos)

    # Largura das barras
    largura_barra = 0.8 / n_metodos
    x = np.arange(n_metricas)

    plt.figure(figsize=(10, 6))
    for i, metodo in enumerate(metodos):
        scores = [dict_scores[metodo].get(m, 0) for m in metricas]
        posicoes = x + i * largura_barra
        plt.bar(posicoes, scores, width=largura_barra, label=metodo)

    plt.xticks(x + largura_barra * (n_metodos - 1) / 2, metricas)
    plt.ylabel("Score")
    plt.title(titulo)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()
