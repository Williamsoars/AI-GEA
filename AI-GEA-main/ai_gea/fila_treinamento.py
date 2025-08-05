import sqlite3
import pickle
import os
from typing import List, Tuple, Dict, Any
import numpy as np

class FilaTreinamento:
    def __init__(self, db_path="fila_treinamento.db"):
        """
        Inicializa a fila de treinamento persistente usando SQLite.
        """
        self.db_path = db_path
        self._criar_tabela()

    def _conectar(self):
        try:
            return sqlite3.connect(self.db_path)
        except sqlite3.Error as e:
            raise RuntimeError(f"Erro ao conectar ao banco de dados: {e}")

    def _criar_tabela(self):
        try:
            with self._conectar() as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS fila (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        grafo BLOB NOT NULL,
                        metricas BLOB NOT NULL,
                        data_adicao TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                conn.commit()
        except sqlite3.Error as e:
            raise RuntimeError(f"Erro ao criar tabela: {e}")

    def adicionar(self, grafo: Any, metricas: Dict[str, Dict[str, float]]) -> None:
        """
        Adiciona um grafo e suas métricas associadas à fila de treinamento.
        """
        try:
            grafo_blob = pickle.dumps(grafo)
            metricas_blob = pickle.dumps(metricas)
            with self._conectar() as conn:
                conn.execute("INSERT INTO fila (grafo, metricas) VALUES (?, ?)", 
                           (grafo_blob, metricas_blob))
                conn.commit()
        except (pickle.PickleError, sqlite3.Error) as e:
            raise RuntimeError(f"Erro ao adicionar à fila: {e}")

    def obter_todos(self) -> List[Tuple[int, Any, Dict]]:
        """
        Retorna todos os grafos e métricas armazenados, ordenados por data de adição.
        """
        try:
            with self._conectar() as conn:
                cursor = conn.execute("SELECT id, grafo, metricas FROM fila ORDER BY data_adicao")
                return [(id_, pickle.loads(g), pickle.loads(m)) 
                        for id_, g, m in cursor.fetchall()]
        except (pickle.PickleError, sqlite3.Error) as e:
            raise RuntimeError(f"Erro ao obter dados da fila: {e}")

    def limpar(self) -> None:
        """
        Remove todos os dados armazenados na fila.
        """
        try:
            with self._conectar() as conn:
                conn.execute("DELETE FROM fila")
                conn.commit()
        except sqlite3.Error as e:
            raise RuntimeError(f"Erro ao limpar fila: {e}")

    def obter_X_y(self, extrair_features_grafo: callable, embeddings: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extrai features (X) e rótulos (y) diretamente da fila, com base nas métricas associadas.

        - X: features estruturais do grafo + métricas por embedding (f1_macro, stress, norma)
        - y: nome do melhor método de embedding (com maior f1_macro)

        Retorna:
        - X: np.ndarray
        - y: np.ndarray
        """
        registros = self.obter_todos()
        X, y = [], []

        for _, grafo, metricas in registros:
            try:
                feats = list(extrair_features_grafo(grafo).values())

                melhores = {
                    metodo: metricas[metodo].get("f1_macro", -1)
                    for metodo in embeddings
                    if metodo in metricas
                }

                if not melhores:
                    continue

                melhor_metodo = max(melhores.items(), key=lambda x: x[1])[0]

                metricas_adicionais = []
                for metodo in embeddings:
                    m = metricas.get(metodo, {})
                    metricas_adicionais.extend([
                        m.get("f1_macro", 0),
                        m.get("stress", 0),
                        m.get("norma", 0)
                    ])

                X.append(feats + metricas_adicionais)
                y.append(melhor_metodo)
            except Exception as e:
                print(f"[AVISO] Registro ignorado por erro: {e}")

        return np.array(X), np.array(y)
