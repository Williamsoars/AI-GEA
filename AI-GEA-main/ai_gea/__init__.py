"""
AI-GEA - Artificial Intelligence for Graph Embedding Analysis
"""
__version__ = "0.1.0"
__author__ = "William Silva"
__email__ = "Williamkauasoaresdasilva@gmail.com"

# Import everything that will be accessible to users directly from the package
from .recommender import EmbeddingRecommender
from .fila_treinamento import FilaTreinamento
from .graph_loader import carregar_grafo
from .features import extrair_features_grafo, benchmark_methods
from .plugins import registrar_metrica, calcular_metricas_personalizadas
from .logger import Logger

from .visual import plot_embedding, plot_comparacao_scores
from .training import treinar_com_fila
from .analysis import avaliar_embeddings, analise_estatistica
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .recommender import EmbeddingRecommender as AI_GEA

# Aliases for convenience
AI_GEA = EmbeddingRecommender

__all__ = [
    'EmbeddingRecommender',
    'EmbeddingRecommenderInferencia',
    'AI_GEA',
    'FilaTreinamento',
    'carregar_grafo',
    'extrair_features_grafo',
    'benchmark_methods',
    'registrar_metrica',
    'calcular_metricas_personalizadas',
    'Logger',
    'calcular_deformacao',
    'plot_embedding',
    'plot_comparacao_scores',
    'treinar_com_fila',
    'avaliar_embeddings',
    'analise_estatistica'
]
