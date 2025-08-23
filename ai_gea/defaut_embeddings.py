import networkx as nx
import numpy as np
from karateclub import Node2Vec, DeepWalk, HOPE, Walklets, NetMF, GraRep
import warnings
warnings.filterwarnings('ignore')

def safe_embedding(G: nx.Graph, model_class, **kwargs) -> dict:
    """Função segura para geração de embeddings com tratamento completo de erros"""
    try:
        # Garante que o grafo tenha labels inteiras e consecutivas começando em 0
        G = nx.convert_node_labels_to_integers(G, first_label=0)
        n_nodes = len(G)
        
        # Verifica se o grafo é muito pequeno para alguns métodos
        if n_nodes < 5:
            return {}
            
        # Ajusta dinamicamente as dimensões baseado no tamanho do grafo
        dimensions = min(kwargs.get('dimensions', 128), n_nodes - 1)
        if dimensions < 2:  # Mínimo de 2 dimensões
            return {}
            
        # Atualiza as dimensões nos kwargs
        kwargs['dimensions'] = dimensions
        
        # Ajustes específicos para cada método
        if model_class == HOPE:
            # HOPE precisa que k seja menor que o número de nós
            k = min(kwargs.get('k', 4), n_nodes - 1)
            kwargs['k'] = max(k, 1)  # k deve ser pelo menos 1
            
        elif model_class == NetMF:
            # NetMF tem restrições adicionais de dimensionalidade
            if dimensions > n_nodes:
                kwargs['dimensions'] = max(n_nodes - 1, 2)
                
        elif model_class == GraRep:
            # GraRep também tem restrições similares
            if dimensions > n_nodes:
                kwargs['dimensions'] = max(n_nodes - 1, 2)
        
        # Cria e treina o modelo
        model = model_class(**kwargs)
        model.fit(G)
        
        # Obtém os embeddings
        embeddings = model.get_embedding()
        
        # Verifica se o embedding é válido
        if not isinstance(embeddings, np.ndarray) or len(embeddings) == 0:
            raise ValueError("Embedding inválido")
            
        # Cria o dicionário de embedding
        return {int(i): np.array(embeddings[i], dtype=np.float32) for i in range(len(embeddings))}
        
    except Exception as e:
        print(f"Erro ao gerar embedding com {model_class.__name__}: {str(e)}")
        return {}

# Definição dos métodos de embedding com parâmetros ajustados
def node2vec_embedding(G: nx.Graph) -> dict:
    n_nodes = len(G)
    dimensions = min(128, max(8, n_nodes // 2))  # Dimensões adaptativas
    return safe_embedding(G, Node2Vec, 
                         dimensions=dimensions,
                         walk_number=10,
                         walk_length=80,
                         window_size=5,
                         epochs=10)

def deepwalk_embedding(G: nx.Graph) -> dict:
    n_nodes = len(G)
    dimensions = min(128, max(8, n_nodes // 2))
    return safe_embedding(G, DeepWalk,
                         dimensions=dimensions,
                         walk_number=10,
                         walk_length=80,
                         window_size=5,
                         epochs=10)

def hope_embedding(G: nx.Graph) -> dict:
    n_nodes = len(G)
    dimensions = min(64, max(4, n_nodes // 3))  # HOPE funciona melhor com menos dimensões
    k = min(4, max(1, n_nodes // 10))  # k adaptativo baseado no tamanho do grafo
    return safe_embedding(G, HOPE,
                         dimensions=dimensions,
                         k=k)

def walklets_embedding(G: nx.Graph) -> dict:
    n_nodes = len(G)
    dimensions = min(128, max(8, n_nodes // 2))
    return safe_embedding(G, Walklets,
                         dimensions=dimensions,
                         walk_number=10,
                         walk_length=80,
                         window_size=5,
                         epochs=10)

def netmf_embedding(G: nx.Graph) -> dict:
    n_nodes = len(G)
    dimensions = min(64, max(4, n_nodes // 3))
    return safe_embedding(G, NetMF,
                         dimensions=dimensions,
                         order=2,  # Reduz a ordem para grafos menores
                         window_size=min(5, n_nodes // 4))

def grarep_embedding(G: nx.Graph) -> dict:
    n_nodes = len(G)
    dimensions = min(64, max(4, n_nodes // 3))
    return safe_embedding(G, GraRep,
                         dimensions=dimensions,
                         order=min(3, n_nodes // 10))  # Ordem adaptativa

# Métodos alternativos robustos
def degree_embedding(G: nx.Graph) -> dict:
    """Embedding baseado em graus dos nós"""
    try:
        G = nx.convert_node_labels_to_integers(G, first_label=0)
        degrees = np.array([d for n, d in G.degree()])
        
        if len(degrees) == 0:
            return {}
        
        # Normaliza e adiciona algumas dimensões extras
        degrees_normalized = degrees / np.max(degrees) if np.max(degrees) > 0 else degrees
        
        # Cria embedding multidimensional simples
        embedding_dim = min(5, len(degrees))
        embeddings = {}
        
        for i, degree in enumerate(degrees_normalized):
            # Cria um vetor baseado no grau com pequenas variações
            base_vector = np.ones(embedding_dim) * degree
            # Adiciona pequeno ruído para diferenciar nós com mesmo grau
            noise = np.random.normal(0, 0.1, embedding_dim)
            embeddings[i] = base_vector + noise
            
        return embeddings
        
    except Exception:
        return {}

def pagerank_embedding(G: nx.Graph) -> dict:
    """Embedding baseado em PageRank"""
    try:
        G = nx.convert_node_labels_to_integers(G, first_label=0)
        pagerank = np.array(list(nx.pagerank(G).values()))
        
        if len(pagerank) == 0:
            return {}
        
        pagerank_normalized = pagerank / np.max(pagerank) if np.max(pagerank) > 0 else pagerank
        
        embedding_dim = min(5, len(pagerank))
        embeddings = {}
        
        for i, pr in enumerate(pagerank_normalized):
            base_vector = np.ones(embedding_dim) * pr
            noise = np.random.normal(0, 0.05, embedding_dim)
            embeddings[i] = base_vector + noise
            
        return embeddings
        
    except Exception:
        return {}

default_embeddings = {
    "deepwalk": deepwalk_embedding,
    "node2vec": node2vec_embedding,
    "walklets": walklets_embedding,
    "hope": hope_embedding,
    "netmf": netmf_embedding,
    "grarep": grarep_embedding,
    "degree": degree_embedding,
    "pagerank": pagerank_embedding
}

# Métodos principais (mais robustos)
robust_embedding_methods = {
    "deepwalk": deepwalk_embedding,
    "node2vec": node2vec_embedding,
    "walklets": walklets_embedding,
    "degree": degree_embedding,
    "pagerank": pagerank_embedding
}

