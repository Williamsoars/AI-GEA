import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np

def plot_embedding(embedding, labels=None, method='TSNE', title='Embedding Visualization'):
    """
    Visualiza um embedding em 2D com redução de dimensionalidade.

    Parâmetros:
    - embedding: np.ndarray (n_nodes, n_features)
    - labels: lista ou array opcional com rótulos dos nós
    - method: 'TSNE' ou 'PCA' (por enquanto apenas TSNE implementado)
    - title: título do gráfico
    """
    if method != 'TSNE':
        raise NotImplementedError("Atualmente, apenas TSNE está implementado.")

    reducer = TSNE(n_components=2, random_state=42, perplexity=30)
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

    plt.title(title)
    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
