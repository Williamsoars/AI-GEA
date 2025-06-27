import networkx as nx
import numpy as np
from karateclub import Node2Vec, DeepWalk, HOPE, Walklets, NetMF, GraRep

def node2vec_embedding(G: nx.Graph) -> dict:
    model = Node2Vec()
    model.fit(G)
    embeddings = model.get_embedding()
    return {node: embeddings[i] for i, node in enumerate(G.nodes())}

def deepwalk_embedding(G: nx.Graph) -> dict:
    model = DeepWalk()
    model.fit(G)
    embeddings = model.get_embedding()
    return {node: embeddings[i] for i, node in enumerate(G.nodes())}

def hope_embedding(G: nx.Graph) -> dict:
    model = HOPE()
    model.fit(G)
    embeddings = model.get_embedding()
    return {node: embeddings[i] for i, node in enumerate(G.nodes())}

def walklets_embedding(G: nx.Graph) -> dict:
    model = Walklets()
    model.fit(G)
    embeddings = model.get_embedding()
    return {node: embeddings[i] for i, node in enumerate(G.nodes())}

def netmf_embedding(G: nx.Graph) -> dict:
    model = NetMF()
    model.fit(G)
    embeddings = model.get_embedding()
    return {node: embeddings[i] for i, node in enumerate(G.nodes())}

def grarep_embedding(G: nx.Graph) -> dict:
    model = GraRep()
    model.fit(G)
    embeddings = model.get_embedding()
    return {node: embeddings[i] for i, node in enumerate(G.nodes())}

default_embeddings = {
    "Node2Vec": node2vec_embedding,
    "DeepWalk": deepwalk_embedding,
    "HOPE": hope_embedding,
    "Walklets": walklets_embedding,
    "NetMF": netmf_embedding,
    "GraRep": grarep_embedding,
}
