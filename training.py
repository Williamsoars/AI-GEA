from tqdm import tqdm
import logging
from .graph_embedding_recommender import EmbeddingRecommender
from .fila_treinamento import FilaTreinamento

def treinar_com_fila(modelo_path="embedding_model.pkl", db_path="fila_treinamento.db"):
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    fila = FilaTreinamento(db_path)
    dados = fila.obter_todos()

    if not dados:
        logger.warning("Nenhum dado na fila para treinamento.")
        return

    logger.info(f"Iniciando treinamento com {len(dados)} grafos...")
    
    try:
        lista_grafos = [grafo for _, grafo, _ in dados]
        resultados_metricas = [metricas for _, _, metricas in dados]

        recommender = EmbeddingRecommender(modelo_path)
        
        with tqdm(total=len(lista_grafos), desc="Treinando modelos") as pbar:
            recommender.treinar(lista_grafos, resultados_metricas)
            pbar.update(len(lista_grafos))

        fila.limpar()
        logger.info(f"Treinamento concluído com sucesso para {len(dados)} grafos.")
    except Exception as e:
        logger.error(f"Erro durante o treinamento: {str(e)}")
        raise
