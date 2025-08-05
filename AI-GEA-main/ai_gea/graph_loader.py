import os
import networkx as nx
from typing import Union

def carregar_grafo(caminho: str) -> Union[nx.Graph, nx.DiGraph]:
    """
    Load a graph from various file formats.
    
    Supported formats: .graphml, .gexf, .gml, .edgelist, .adjlist
    
    Args:
        caminho: Path to graph file
        
    Returns:
        NetworkX graph object
        
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If format is not supported
        RuntimeError: If there's an error loading the graph
    """
    if not os.path.exists(caminho):
        raise FileNotFoundError(f"File not found: {caminho}")
    
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
            raise ValueError(f'Unsupported graph format: {ext}')
    except Exception as e:
        raise RuntimeError(f"Error loading graph from file {caminho}: {str(e)}")
import os
import networkx as nx
from typing import Union

def carregar_grafo(caminho: str) -> Union[nx.Graph, nx.DiGraph]:
    """
    Load a graph from various file formats.
    
    Supported formats: .graphml, .gexf, .gml, .edgelist, .adjlist
    
    Args:
        caminho: Path to graph file
        
    Returns:
        NetworkX graph object
        
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If format is not supported
        RuntimeError: If there's an error loading the graph
    """
    if not os.path.exists(caminho):
        raise FileNotFoundError(f"File not found: {caminho}")
    
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
            raise ValueError(f'Unsupported graph format: {ext}')
    except Exception as e:
        raise RuntimeError(f"Error loading graph from file {caminho}: {str(e)}")

import os
import networkx as nx
import pandas as pd

def load_cora_graph(data_dir: str = "data/cora") -> nx.Graph:
    """
    Carrega o grafo Cora a partir dos arquivos .cites e .content.
    
    Args:
        data_dir: Caminho para a pasta onde estão os arquivos cora.content e cora.cites
        
    Returns:
        G: grafo NetworkX com atributos 'label' e 'features' nos nós
    """
    # Arquivos
    content_path = os.path.join(data_dir, "cora.content")
    cites_path = os.path.join(data_dir, "cora.cites")
    
    # Verificação
    if not os.path.exists(content_path) or not os.path.exists(cites_path):
        raise FileNotFoundError("Arquivos 'cora.content' e/ou 'cora.cites' não encontrados.")

    # Carregar conteúdo
    content_df = pd.read_csv(content_path, sep='\t', header=None)
    features = content_df.iloc[:, 1:-1].values
    labels = content_df.iloc[:, -1].values
    node_ids = content_df.iloc[:, 0].values

    # Mapeamento node_id → index
    id_map = {node_id: i for i, node_id in enumerate(node_ids)}

    # Criar grafo
    G = nx.Graph()

    # Adicionar nós com atributos
    for i, node_id in enumerate(node_ids):
        G.add_node(i, label=labels[i], features=features[i])

    # Adicionar arestas
    with open(cites_path, 'r') as f:
        for line in f:
            src, dst = line.strip().split()
            if src in id_map and dst in id_map:
                G.add_edge(id_map[src], id_map[dst])

    return G
