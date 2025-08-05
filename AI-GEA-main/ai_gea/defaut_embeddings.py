import networkx as nx
import numpy as np
from karateclub import Node2Vec, DeepWalk, HOPE, Walklets, NetMF, GraRep

def safe_embedding(G: nx.Graph, model_class) -> dict:
    """Função segura para geração de embeddings com tratamento completo de erros"""
    try:
        # Garante que o grafo tenha labels inteiras e consecutivas começando em 0
        G = nx.convert_node_labels_to_integers(G, first_label=0)
        
        # Cria e treina o modelo
        model = model_class()
        model.fit(G)
        
        # Obtém os embeddings (já é um numpy array)
        embeddings = model.get_embedding()
        
        # Verifica se o formato está correto
        if not isinstance(embeddings, np.ndarray):
            raise ValueError("Embedding não é um numpy array")
            
        # Cria o dicionário de embedding garantindo os tipos corretos
        return {int(i): np.array(embeddings[i], dtype=np.float32) for i in range(len(embeddings))}
        
    except Exception as e:
        print(f"Erro ao gerar embedding com {model_class.__name__}: {str(e)}")
        return {}

# Definição dos métodos de embedding usando a função segura
def node2vec_embedding(G: nx.Graph) -> dict:
    return safe_embedding(G, Node2Vec)

def deepwalk_embedding(G: nx.Graph) -> dict:
    return safe_embedding(G, DeepWalk)

def hope_embedding(G: nx.Graph) -> dict:
    return safe_embedding(G, HOPE)

def walklets_embedding(G: nx.Graph) -> dict:
    return safe_embedding(G, Walklets)

def netmf_embedding(G: nx.Graph) -> dict:
    return safe_embedding(G, NetMF)

def grarep_embedding(G: nx.Graph) -> dict:
    return safe_embedding(G, GraRep)

default_embeddings = {
    "deepwalk": deepwalk_embedding,
    "walklets": walklets_embedding,
    "hope": hope_embedding,
    "node2vec": node2vec_embedding,
    "netmf": netmf_embedding,
    "grarep": grarep_embedding
}
