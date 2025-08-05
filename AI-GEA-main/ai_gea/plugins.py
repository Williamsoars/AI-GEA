
import networkx as nx
import numpy as np
from typing import Dict, Callable, TypeVar, Any

# Dicionário para armazenar métricas personalizadas
custom_metrics: Dict[str, Callable[[nx.Graph, np.ndarray], Any]] = {}
T = TypeVar('T')  # Para tipos genéricos na métrica

# Tipo específico para funções de métrica
MetricFunction = Callable[[nx.Graph, np.ndarray], T]

custom_metrics: Dict[str, MetricFunction] = {}

def registrar_metrica(nome: str, funcao: MetricFunction) -> None:
    """Registra uma nova métrica personalizada.
    
    Args:
        nome: Nome único para a métrica
        funcao: Função que recebe (G, embedding) e retorna um valor
    """
    if not callable(funcao):
        raise ValueError("A função da métrica deve ser chamável")
    custom_metrics[nome] = funcao

def calcular_metricas_personalizadas(G: nx.Graph, embedding: np.ndarray) -> Dict[str, Any]:
    """Calcula todas as métricas personalizadas registradas.
    
    Args:
        G: Grafo de entrada
        embedding: Embedding do grafo
        
    Returns:
        Dicionário com os resultados das métricas
    """
    resultados = {}
    for nome, funcao in custom_metrics.items():
        try:
            resultados[nome] = funcao(G, embedding)
        except Exception as e:
            print(f"Erro ao calcular métrica '{nome}': {str(e)}")
            resultados[nome] = None
    return resultados
