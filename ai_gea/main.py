import networkx as nx
from graph_embedding_recommender.recommender import EmbeddingRecommender
from graph_embedding_evaluator.evaluation import avaliar_metodos
from graph_embedding_recommender.logger import Logger

def avaliar_grafos_base():
    grafos = [
        nx.karate_club_graph(),
        nx.cycle_graph(32),
        nx.path_graph(32),
    ]
    return [
        avaliar_metodos(G, ["Node2Vec", "DeepWalk", "LINE"], n_execucoes=5)
        for G in grafos
    ]

def main():
    try:
        # Treinamento
        grafos = avaliar_grafos_base()
        resultados = [resultado for resultado in grafos]
        
        ia = EmbeddingRecommender()
        ia.treinar(grafos, resultados)

        # Recomendação
        G_novo = nx.balanced_tree(2, 4)
        metodo_recomendado, scores = ia.recomendar(G_novo)
        
        print("Método recomendado:", metodo_recomendado)
        print("Scores previstos:", scores)

        # Log
        logger = Logger()
        logger.log("BalancedTree", scores, metodo_recomendado)
        
    except Exception as e:
        print(f"Erro durante a execução: {str(e)}")
        raise

if __name__ == "__main__":
    main()
