import numpy as np
import pickle
import os
from typing import Dict, Tuple
from .features import extrair_features_grafo
from .fila_treinamento import FilaTreinamento
from sklearn.model_selection import KFold
import numpy as np
class EmbeddingRecommenderInferencia:
    registered_embedding_methods = {}
    def get_methods()
        return registered_embedding_methods

    def register_embedding_method(name, func):
        """
        Registra um novo método de embedding com nome e função.

        Parâmetros:
        - name (str): nome do método (ex: "MeuNode2Vec")
        - func (callable): função que recebe um grafo G e retorna um embedding (np.ndarray)
        """
        registered_embedding_methods[name] = func
    def cross_validate(grafos, methods, k_folds=5, avaliar_metodos_fn=None):
        """
        Executa validação cruzada dos métodos de embedding sobre os grafos fornecidos.

        Parâmetros:
        - grafos: lista de grafos (NetworkX)
        - methods: lista de nomes de métodos (ex: ["Node2Vec", "DeepWalk"])
        - k_folds: número de divisões
        - avaliar_metodos_fn: função que avalia um grafo (deve aceitar: G, methods)

        Retorna:
        - dicionário com médias e desvios padrão por método.
        """
        if avaliar_metodos_fn is None:
            raise ValueError("É necessário passar a função 'avaliar_metodos_fn' para avaliação dos grafos.")

        kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
        scores = {method: [] for method in methods}

        grafos = np.array(grafos)

        for train_idx, test_idx in kf.split(grafos):
            for i in test_idx:
                G = grafos[i]
                resultados = avaliar_metodos_fn(G, methods)
                for metodo, metricas in resultados.items():
                    if isinstance(metricas, dict) and "f1_macro" in metricas:
                        scores[metodo].append(metricas["f1_macro"])  # ou outra métrica desejada

        # Calcular médias e desvios
        resumo = {
            metodo: {
                "media": np.mean(vals),
                "desvio": np.std(vals)
            }
            for metodo, vals in scores.items()
        }

        return resumo
    def __init__(self, modelo_path="embedding_model.pkl", db_path="fila_treinamento.db"):
        self.modelo_path = modelo_path
        self.db_path = db_path
        self.embeddings = ["Node2Vec", "DeepWalk", "LINE", "HOPE", "Walklets", "NetMF", "GraRep"]
        self._load_model()
        self.fila = FilaTreinamento(db_path=self.db_path)

    def _load_model(self):
        if not os.path.exists(self.modelo_path):
            raise FileNotFoundError(f"Arquivo de modelo não encontrado: {self.modelo_path}")
            
        try:
            with open(self.modelo_path, "rb") as f:
                self.modelos = pickle.load(f)
                
            # Verifica se todos os embeddings têm modelos
            for emb in self.embeddings:
                if emb not in self.modelos:
                    raise ValueError(f"Modelo para embedding '{emb}' não encontrado")
        except (pickle.PickleError, EOFError) as e:
            raise RuntimeError(f"Erro ao carregar modelo: {str(e)}")
    def recomendar(self, G, metricas_resultantes=None) -> Tuple[str, Dict[str, float]]:
        try:
            feats = np.array(list(extrair_features_grafo(G).values())).reshape(1, -1)
            scores = {emb: self.modelos[emb].predict(feats)[0] for emb in self.embeddings}
            recomendado = max(scores.items(), key=lambda x: x[1])

            if metricas_resultantes:
                self.fila.adicionar(G, metricas_resultantes)

            return recomendado[0], scores
        except Exception as e:
            raise RuntimeError(f"Erro durante a recomendação: {str(e)}")
