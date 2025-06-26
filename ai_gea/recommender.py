from typing import Dict, Tuple, List, Callable, Optional, Union
import numpy as np
import pickle
import os
from sklearn.base import BaseEstimator  # Adicione esta importação

class EmbeddingRecommender:
    def __init__(self, modelo_path: str = "embedding_model.pkl", db_path: str = "fila_treinamento.db"):
        self.modelo_path = modelo_path
        self.db_path = db_path
        self.embeddings: List[str] = ["Node2Vec", "DeepWalk", "LINE", "HOPE", "Walklets", "NetMF", "GraRep"]
        self.modelos: Dict[str, BaseEstimator] = {}  # Tipo explícito para modelos
        self._load_model()
        self.fila = FilaTreinamento(db_path=self.db_path)

    def recomendar(self, G: nx.Graph, metricas_resultantes: Optional[Dict[str, Dict[str, float]]] = None) -> Tuple[str, Dict[str, float]]:
        """Método com type hints completos"""
        try:
            feats = np.array(list(extrair_features_grafo(G).values())).reshape(1, -1)
            scores = {emb: self.modelos[emb].predict(feats)[0] for emb in self.embeddings}
            recomendado = max(scores.items(), key=lambda x: x[1])

            if metricas_resultantes:
                self.fila.adicionar(G, metricas_resultantes)

            return recomendado[0], scores
        except Exception as e:
            raise RuntimeError(f"Error during recommendation: {str(e)}")

    def treinar(self, grafos: List, resultados: List[Dict]) -> None:
        """Train the recommendation model"""
        # Implementation here
        pass
    def _load_model(self) -> None:
        if not os.path.exists(self.modelo_path):
            raise FileNotFoundError(f"Model file not found: {self.modelo_path}")
            
        try:
            with open(self.modelo_path, "rb") as f:
                self.modelos = pickle.load(f)
                
            # Verify all embeddings have models
            for emb in self.embeddings:
                if emb not in self.modelos:
                    raise ValueError(f"Model for embedding '{emb}' not found")
        except (pickle.PickleError, EOFError) as e:
            raise RuntimeError(f"Error loading model: {str(e)}")
      def salvar_modelo(self, caminho: str) -> None:
        """Save the trained model"""
        with open(caminho, 'wb') as f:
            pickle.dump(self.modelos, f)

class EmbeddingRecommenderInferencia:
    registered_embedding_methods: Dict[str, Callable] = {}

    @classmethod
    def get_methods(cls) -> Dict[str, Callable]:
        """Get all registered embedding methods"""
        return cls.registered_embedding_methods

    @classmethod
    def register_embedding_method(cls, name: str, func: Callable) -> None:
        """
        Register a new embedding method with name and function.

        Parameters:
        - name (str): method name (e.g., "MyNode2Vec")
        - func (callable): function that receives a graph G and returns an embedding (np.ndarray)
        """
        if not callable(func):
            raise ValueError("Function must be callable")
        cls.registered_embedding_methods[name] = func

    @classmethod
    def cross_validate(cls, grafos: List, methods: List[str], k_folds: int = 5, 
                      avaliar_metodos_fn: Callable = None) -> Dict[str, Dict[str, float]]:
        """
        Perform cross-validation of embedding methods on provided graphs.

        Parameters:
        - grafos: list of graphs (NetworkX)
        - methods: list of method names (e.g., ["Node2Vec", "DeepWalk"])
        - k_folds: number of splits
        - avaliar_metodos_fn: function that evaluates a graph (should accept: G, methods)

        Returns:
        - dictionary with means and standard deviations per method.
        """
        if avaliar_metodos_fn is None:
            raise ValueError("The 'avaliar_metodos_fn' function is required for graph evaluation.")

        kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
        scores = {method: [] for method in methods}

        grafos = np.array(grafos)

        for train_idx, test_idx in kf.split(grafos):
            for i in test_idx:
                G = grafos[i]
                resultados = avaliar_metodos_fn(G, methods)
                for metodo, metricas in resultados.items():
                    if isinstance(metricas, dict) and "f1_macro" in metricas:
                        scores[metodo].append(metricas["f1_macro"])

        # Calculate means and standard deviations
        resumo = {
            metodo: {
                "media": np.mean(vals),
                "desvio": np.std(vals)
            }
            for metodo, vals in scores.items()
        }

        return resumo
