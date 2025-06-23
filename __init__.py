"""
AI-GEA - Artificial Intelligence for Graph Embedding Analysis

Um sistema avançado para análise e recomendação de métodos de embedding para grafos.
"""
__version__ = "0.1.0"
__author__ = "William Silva"
__email__ = "Williamkauasoaresdasilva@gmail.com"
__license__ = "MIT"

# Importações principais
from .recomendador import EmbeddingRecommender, EmbeddingRecommenderInferencia
from .fila_treinamento import FilaTreinamento
from .training import treinar_com_fila
from .features import extrair_features_grafo
from .graph_loader import carregar_grafo
from .logger import Logger
from .plugins import registrar_metrica, calcular_metricas_personalizadas

# Suporte para importação direta do nome principal
AI_GEA = EmbeddingRecommender  # Alias para o principal

__all__ = [
    'AI_GEA',
    'EmbeddingRecommender',
    'EmbeddingRecommenderInferencia',
    'FilaTreinamento',
    'treinar_com_fila',
    'extrair_features_grafo',
    'carregar_grafo',
    'Logger',
    'registrar_metrica',
    'calcular_metricas_personalizadas',
]
