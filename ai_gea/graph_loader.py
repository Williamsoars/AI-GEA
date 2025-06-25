import os
import networkx as nx

def carregar_grafo(caminho):
    if not os.path.exists(caminho):
        raise FileNotFoundError(f"Arquivo não encontrado: {caminho}")
    
    ext = os.path.splitext(caminho)[-1].lower()
    try:
        if ext == '.graphml':
            return nx.read_graphml(caminho)
        elif ext == '.gexf':
            return nx.read_gexf(caminho)
        elif ext == '.gml':
            return nx.read_gml(caminho)
        elif ext == '.edgelist':
            return nx.read_edgelist(caminho)
        elif ext == '.adjlist':
            return nx.read_adjlist(caminho)
        else:
            raise ValueError(f'Formato de grafo não suportado: {ext}')
    except Exception as e:
        raise RuntimeError(f"Erro ao carregar grafo do arquivo {caminho}: {str(e)}")
