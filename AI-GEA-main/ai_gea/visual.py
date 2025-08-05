import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import numpy as np
import seaborn as sns
from typing import Dict
import pandas as pd

def plot_embedding(embedding, labels=None, method='TSNE', title='Embedding Visualization', ax=None):
    """
    Enhanced embedding visualization with more customization options.
    """
    if method == 'TSNE':
        reducer = TSNE(n_components=2, random_state=42, perplexity=30)
    elif method == 'PCA':
        reducer = PCA(n_components=2)
    else:
        raise ValueError("Método inválido. Use 'TSNE' ou 'PCA'.")

    X = np.array(list(embedding.values())) if isinstance(embedding, dict) else np.array(embedding)
    reduced = reducer.fit_transform(X)

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    
    if labels is not None:
        labels = np.array(labels)
        unique_labels = np.unique(labels)
        for label in unique_labels:
            idx = labels == label
            ax.scatter(reduced[idx, 0], reduced[idx, 1], label=f"Label {label}", s=30)
        ax.legend()
    else:
        ax.scatter(reduced[:, 0], reduced[:, 1], s=30)

    ax.set_title(f"{title} ({method})")
    ax.set_xlabel("Dim 1")
    ax.set_ylabel("Dim 2")
    ax.grid(True)
    
    if ax is None:
        plt.tight_layout()
        plt.show()

def plot_comparacao_scores(dict_scores: Dict[str, Dict[str, float]], 
                         titulo: str = 'Comparação de Scores por Método',
                         figsize: tuple = (12, 6)):
    """
    Enhanced score comparison plot with more customization.
    """
    if not dict_scores:
        print("Nenhum dado fornecido para plotagem.")
        return

    metricas = sorted({m for valores in dict_scores.values() for m in valores.keys()})
    metodos = list(dict_scores.keys())
    n_metricas = len(metricas)
    
    fig, ax = plt.subplots(figsize=figsize)
    width = 0.8 / len(metodos)
    x = np.arange(n_metricas)
    
    for i, metodo in enumerate(metodos):
        scores = [dict_scores[metodo].get(m, 0) for m in metricas]
        ax.bar(x + i*width, scores, width=width, label=metodo)
    
    ax.set_xticks(x + width*(len(metodos)-1)/2)
    ax.set_xticklabels(metricas, rotation=45)
    ax.set_ylabel("Score")
    ax.set_title(titulo)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    fig.tight_layout()
    plt.show()

def plot_feature_importance(features: list, importances: np.ndarray, 
                          top_n: int = 15, title: str = "Feature Importance"):
    """
    Plot feature importance from a trained model.
    """
    indices = np.argsort(importances)[-top_n:]
    
    plt.figure(figsize=(10, 6))
    plt.title(title)
    plt.barh(range(top_n), importances[indices], align='center')
    plt.yticks(range(top_n), [features[i] for i in indices])
    plt.xlabel("Importance Score")
    plt.tight_layout()
    plt.show()

def plot_correlation_matrix(df: pd.DataFrame, method: str = 'pearson'):
    """
    Plot a correlation matrix for all features and metrics.
    """
    plt.figure(figsize=(14, 12))
    corr = df.corr(method=method)
    mask = np.triu(np.ones_like(corr, dtype=bool))
    
    sns.heatmap(corr, mask=mask, cmap='coolwarm', center=0,
               annot=True, fmt=".2f", square=True,
               linewidths=.5, cbar_kws={"shrink": .8})
    
    plt.title(f"Matriz de Correlação ({method.capitalize()})")
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()
