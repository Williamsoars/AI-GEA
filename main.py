import networkx as nx
from graph_embedding_recommender.recommender import EmbeddingRecommender
from graph_embedding_evaluator.evaluation import avaliar_metodos

grafos = [
    nx.karate_club_graph(),
    nx.cycle_graph(32),
    nx.path_graph(32),
]
resultados = [
    avaliar_metodos(G, ["Node2Vec", "DeepWalk", "LINE"], n_execucoes=5)
    for G in grafos
]

ia = EmbeddingRecommender()
ia.treinar(grafos, resultados)

G_novo = nx.balanced_tree(2, 4)
metodo_recomendado, scores = ia.recomendar(G_novo)
print("Método recomendado:", metodo_recomendado)
print("Scores previstos:", scores)


from graph_embedding_recommender.logger import Logger
logger = Logger()
logger.log("BalancedTree", scores, metodo_recomendado)
