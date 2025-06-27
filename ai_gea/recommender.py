from typing import Dict, Tuple, List, Callable, Optional
import numpy as np
import pickle
import os
import networkx as nx
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

class EmbeddingRecommender:
    def __init__(self, modelo_path: str = "embedding_model.pkl", db_path: str = "fila_treinamento.db"):
        """
        Inicializa o sistema de recomendação de embeddings.
        """
        self.modelo_path = modelo_path
        self.db_path = db_path
        self.embeddings: List[str] = ["Node2Vec", "DeepWalk", "LINE", "HOPE", "Walklets", "NetMF", "GraRep"]
        self.modelos: Dict[str, BaseEstimator] = {}
        self._load_model()
        self.fila = FilaTreinamento(db_path=self.db_path)

    def recomendar(self, G: nx.Graph, metricas_resultantes: Optional[Dict[str, Dict[str, float]]] = None) -> Tuple[str, Dict[str, float]]:
        """
        Retorna o melhor método de embedding com base nas features do grafo e nas métricas (f1, stress, norma...).

        Parâmetros:
        - G: grafo de entrada
        - metricas_resultantes: dicionário com métricas por método (opcional)

        Retorna:
        - nome do método recomendado
        - dicionário com probabilidades por método
        """
        try:
            feats = list(extrair_features_grafo(G).values())

            # Features adicionais com métricas
            metricas = []
            for metodo in self.embeddings:
                m = metricas_resultantes.get(metodo, {}) if metricas_resultantes else {}
                metricas.extend([
                    m.get("f1_macro", 0),
                    m.get("stress", 0),
                    m.get("norma", 0)
                ])

            entrada = np.array(feats + metricas).reshape(1, -1)
            modelo = self.modelos["classificador"]
            pred = modelo.predict(entrada)[0]
            probas = modelo.predict_proba(entrada)[0]
            scores = dict(zip(modelo.classes_, probas))

            if metricas_resultantes:
                self.fila.adicionar(G, metricas_resultantes)

            return pred, scores

        except Exception as e:
            raise RuntimeError(f"Erro na recomendação: {str(e)}")

    def treinar(self, grafos: List[nx.Graph], resultados: List[Dict[str, Dict[str, float]]]) -> None:
        """
        Treina um classificador para prever o melhor método de embedding.

        Parâmetros:
        - grafos: lista de grafos
        - resultados: lista de dicionários com métricas para cada método em cada grafo
        """
        if len(grafos) != len(resultados):
            raise ValueError("O número de grafos deve ser igual ao número de entradas de resultados.")

        X, y = [], []

        for G, metrica_dict in zip(grafos, resultados):
            feats = list(extrair_features_grafo(G).values())

            melhores = {
                metodo: metricas.get("f1_macro", -1)
                for metodo, metricas in metrica_dict.items()
                if "f1_macro" in metricas
            }

            if not melhores:
                continue  # Ignora grafos sem F1 válido

            melhor_metodo = max(melhores.items(), key=lambda x: x[1])[0]

            metricas_adicionais = []
            for metodo in self.embeddings:
                m = metrica_dict.get(metodo, {})
                metricas_adicionais.extend([
                    m.get("f1_macro", 0),
                    m.get("stress", 0),
                    m.get("norma", 0)
                ])

            X.append(feats + metricas_adicionais)
            y.append(melhor_metodo)

        X = np.array(X)
        y = np.array(y)

        clf = Pipeline([
            ("scaler", StandardScaler()),
            ("modelo", RandomForestClassifier(n_estimators=200, random_state=42))
        ])

        clf.fit(X, y)
        self.modelos = {"classificador": clf}
        self.salvar_modelo(self.modelo_path)

    def cross_validate_modelo(self, grafos: List[nx.Graph], resultados: List[Dict[str, Dict[str, float]]], folds: int = 5) -> str:
        """
        Avalia o modelo por validação cruzada estratificada.

        Parâmetros:
        - grafos: lista de grafos
        - resultados: lista de métricas por grafo
        - folds: número de partições (k-fold)

        Retorna:
        - Relatório completo de classificação por fold.
        """
        X, y = [], []

        for G, metrica_dict in zip(grafos, resultados):
            feats = list(extrair_features_grafo(G).values())

            melhores = {
                metodo: metricas.get("f1_macro", -1)
                for metodo, metricas in metrica_dict.items()
                if "f1_macro" in metricas
            }

            if not melhores:
                continue

            melhor_metodo = max(melhores.items(), key=lambda x: x[1])[0]

            metricas_adicionais = []
            for metodo in self.embeddings:
                m = metrica_dict.get(metodo, {})
                metricas_adicionais.extend([
                    m.get("f1_macro", 0),
                    m.get("stress", 0),
                    m.get("norma", 0)
                ])

            X.append(feats + metricas_adicionais)
            y.append(melhor_metodo)

        X = np.array(X)
        y = np.array(y)

        clf = Pipeline([
            ("scaler", StandardScaler()),
            ("modelo", RandomForestClassifier(n_estimators=200, random_state=42))
        ])

        skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
        y_true_all, y_pred_all = [], []

        for train_idx, test_idx in skf.split(X, y):
            clf.fit(X[train_idx], y[train_idx])
            y_pred = clf.predict(X[test_idx])
            y_true = y[test_idx]
            y_pred_all.extend(y_pred)
            y_true_all.extend(y_true)

        return classification_report(y_true_all, y_pred_all, digits=4)

    def _load_model(self) -> None:
        """
        Carrega o modelo salvo do disco, se existir.
        """
        if not os.path.exists(self.modelo_path):
            return  # Nenhum modelo treinado ainda

        try:
            with open(self.modelo_path, "rb") as f:
                self.modelos = pickle.load(f)
            if "classificador" not in self.modelos:
                raise ValueError("Modelo carregado não contém 'classificador'")
        except (pickle.PickleError, EOFError) as e:
            raise RuntimeError(f"Erro ao carregar modelo: {str(e)}")

    def salvar_modelo(self, caminho: str) -> None:
        """
        Salva o modelo no caminho especificado.
        """
        with open(caminho, 'wb') as f:
            pickle.dump(self.modelos, f)

