import networkx as nx
from .default_embeddings import default_embeddings
from .evaluation import avaliar_metodos
from .recommender import Recommender
from .logger import Logger

def avaliar_grafos_base():
    grafos = [
        nx.karate_club_graph(),
        nx.cycle_graph(32),
        nx.path_graph(32),
    ]
    metodos = list(default_embeddings.keys())
    resultados = [
        avaliar_metodos(G, metodos, default_embeddings, labels_true=nx.get_node_attributes(G, 'club'), n_execucoes=3)
        for G in grafos
    ]
    return grafos, resultados

def main():
    try:
        # Treinamento
        grafos, resultados = avaliar_grafos_base()
        
        ia = Recommender()
        ia.treinar(grafos, resultados)

        # Recomendação para novo grafo
        G_novo = nx.balanced_tree(2, 4)
        metodos = list(default_embeddings.keys())
        metricas_novo = avaliar_metodos(G_novo, metodos, default_embeddings, labels_true=None, n_execucoes=3)
        metodo_recomendado, scores = ia.recomendar(G_novo, metricas_resultantes=metricas_novo)

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
