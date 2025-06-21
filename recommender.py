import os
import pickle
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from .features import extrair_features_grafo

class EmbeddingRecommender:
    def __init__(self, modelo_path="embedding_model.pkl"):
        self.modelo_path = modelo_path
        self.modelos = {}
        self.embeddings = ["Node2Vec", "DeepWalk", "LINE", "HOPE", "Walklets", "NetMF", "GraRep"]
        self._load_model()

    def _load_model(self):
        if os.path.exists(self.modelo_path):
            with open(self.modelo_path, "rb") as f:
                self.modelos = pickle.load(f)
        else:
            self.modelos = {emb: Pipeline([
                ("scaler", StandardScaler()),
                ("regressor", RandomForestRegressor(n_estimators=100))
            ]) for emb in self.embeddings}

    def _save_model(self):
        with open(self.modelo_path, "wb") as f:
            pickle.dump(self.modelos, f)

    def treinar(self, lista_grafos, resultados_metricas):
        features, targets = {emb: [] for emb in self.embeddings}, {emb: [] for emb in self.embeddings}
        for G, metricas_por_metodo in zip(lista_grafos, resultados_metricas):
            feats = extrair_features_grafo(G)
            for emb in self.embeddings:
                if emb in metricas_por_metodo:
                    features[emb].append(list(feats.values()))
                    m = metricas_por_metodo[emb]
                    score = -m["stress"] + m["f1"] + m["auc"] - m.get("deformacao", 0)
                    targets[emb].append(score)
        for emb in self.embeddings:
            if features[emb]:
                self.modelos[emb].fit(features[emb], targets[emb])
        self._save_model()

    def recomendar(self, G):
        feats = np.array(list(extrair_features_grafo(G).values())).reshape(1, -1)
        scores = {emb: self.modelos[emb].predict(feats)[0] for emb in self.embeddings}
        recomendado = max(scores.items(), key=lambda x: x[1])
        return recomendado[0], scores
