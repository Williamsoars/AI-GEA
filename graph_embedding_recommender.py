import numpy as np
import pickle
import os
from typing import Dict, Tuple
from .features import extrair_features_grafo
from .fila_treinamento import FilaTreinamento

class EmbeddingRecommenderInferencia:
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
