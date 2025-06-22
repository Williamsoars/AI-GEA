import sqlite3
import pickle
import os

class FilaTreinamento:
    def __init__(self, db_path="fila_treinamento.db"):
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

    def adicionar(self, grafo, metricas):
        try:
            grafo_blob = pickle.dumps(grafo)
            metricas_blob = pickle.dumps(metricas)
            with self._conectar() as conn:
                conn.execute("INSERT INTO fila (grafo, metricas) VALUES (?, ?)", 
                           (grafo_blob, metricas_blob))
                conn.commit()
        except (pickle.PickleError, sqlite3.Error) as e:
            raise RuntimeError(f"Erro ao adicionar à fila: {e}")

    def obter_todos(self):
        try:
            with self._conectar() as conn:
                cursor = conn.execute("SELECT id, grafo, metricas FROM fila ORDER BY data_adicao")
                return [(id_, pickle.loads(g), pickle.loads(m)) 
                        for id_, g, m in cursor.fetchall()]
        except (pickle.PickleError, sqlite3.Error) as e:
            raise RuntimeError(f"Erro ao obter dados da fila: {e}")

    def limpar(self):
        try:
            with self._conectar() as conn:
                conn.execute("DELETE FROM fila")
                conn.commit()
        except sqlite3.Error as e:
            raise RuntimeError(f"Erro ao limpar fila: {e}")
