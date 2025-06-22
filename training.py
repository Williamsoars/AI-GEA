from .recomendador import EmbeddingRecommender
from .fila_treinamento import FilaTreinamento

def treinar_com_fila(modelo_path="embedding_model.pkl", db_path="fila_treinamento.db"):
    fila = FilaTreinamento(db_path)
    dados = fila.obter_todos()

    if not dados:
        print("Nenhum dado na fila.")
        return

    lista_grafos = [grafo for _, grafo, _ in dados]
    resultados_metricas = [metricas for _, _, metricas in dados]

    recommender = EmbeddingRecommender(modelo_path)
    recommender.treinar(lista_grafos, resultados_metricas)

    fila.limpar()
    print(f"Treinamento concluído com {len(dados)} grafos.")
