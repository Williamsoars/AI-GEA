"""
AI-GEA - Artificial Intelligence for Graph Embedding Analysis
"""
__version__ = "0.1.0"
__author__ = "William Silva"
__email__ = "Williamkauasoaresdasilva@gmail.com"

# Importe TUDO que será acessível ao usuário diretamente do pacote
from .recommender import EmbeddingRecommender, EmbeddingRecommenderInferencia
from .fila_treinamento import FilaTreinamento
from .graph_loader import carregar_grafo
from .features import extrair_features_grafo
from .plugins import registrar_metrica, calcular_metricas_personalizadas
from .logger import Logger

# Aliases para facilitar
AI_GEA = EmbeddingRecommender

__all__ = [
    'EmbeddingRecommender',
    'EmbeddingRecommenderInferencia',
    'AI_GEA',
    'FilaTreinamento',
    'carregar_grafo',
    'extrair_features_grafo',
    'registrar_metrica',
    'calcular_metricas_personalizadas',
    'Logger'
]
