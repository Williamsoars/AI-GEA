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
