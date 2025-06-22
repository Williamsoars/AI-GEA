# arquivo: fila_treinamento.py
import sqlite3
import pickle
import os

class FilaTreinamento:
    def __init__(self, db_path="fila_treinamento.db"):
        self.db_path = db_path
        self._criar_tabela()

    def _conectar(self):
        return sqlite3.connect(self.db_path)

    def _criar_tabela(self):
        with self._conectar() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS fila (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    grafo BLOB NOT NULL,
                    metricas BLOB NOT NULL
                )
            """)
            conn.commit()

    def adicionar(self, grafo, metricas):
        grafo_blob = pickle.dumps(grafo)
        metricas_blob = pickle.dumps(metricas)
        with self._conectar() as conn:
            conn.execute("INSERT INTO fila (grafo, metricas) VALUES (?, ?)", (grafo_blob, metricas_blob))
            conn.commit()

    def obter_todos(self):
        with self._conectar() as conn:
            cursor = conn.execute("SELECT id, grafo, metricas FROM fila")
            dados = [(id_, pickle.loads(g), pickle.loads(m)) for id_, g, m in cursor.fetchall()]
        return dados

    def limpar(self):
        with self._conectar() as conn:
            conn.execute("DELETE FROM fila")
            conn.commit()
