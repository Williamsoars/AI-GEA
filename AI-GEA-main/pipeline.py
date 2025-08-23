import numpy as np
import pandas as pd
import networkx as nx
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from ai_gea import EmbeddingRecommender
from ai_gea.defaut_embeddings import default_embeddings
from ai_gea.features import extrair_features_grafo
from ai_gea.evaluation import calcular_stress, reconstruction_error
import warnings
from tqdm import tqdm
import json
import os
from datetime import datetime

warnings.filterwarnings('ignore')

class EmbeddingPredictionExperiment:
    def __init__(self, n_runs=10, test_size=0.2, rf_params=None, database_path="embedding_benchmark.db"):
        self.n_runs = n_runs
        self.test_size = test_size
        self.database_path = database_path
        self.metricas = {
            'stress': calcular_stress,
            'reconstruction_error': reconstruction_error,
            'f1_macro': self.calcular_f1_macro
        }
        
        # Parâmetros do Random Forest (personalizáveis)
        self.rf_params = rf_params or {
            'n_estimators': 200,
            'max_depth': 10,
            'random_state': 42,
            'min_samples_split': 2,
            'min_samples_leaf': 1
        }
        
        self.embedding_methods = list(default_embeddings.keys())
        self.results = []
        self.model = None
        self._initialize_database()

    def _initialize_database(self):
        """Inicializa o banco de dados SQLite para armazenar resultados"""
        import sqlite3
        
        self.conn = sqlite3.connect(self.database_path)
        cursor = self.conn.cursor()
        
        # Tabela de grafos
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS graphs (
                graph_id INTEGER PRIMARY KEY AUTOINCREMENT,
                graph_type TEXT,
                n_nodes INTEGER,
                n_edges INTEGER,
                density REAL,
                features TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Tabela de resultados de embeddings
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS embedding_results (
                result_id INTEGER PRIMARY KEY AUTOINCREMENT,
                graph_id INTEGER,
                embedding_method TEXT,
                stress REAL,
                reconstruction_error REAL,
                f1_macro REAL,
                execution_time REAL,
                FOREIGN KEY (graph_id) REFERENCES graphs (graph_id)
            )
        ''')
        
        # Tabela de recomendações
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS recommendations (
                recommendation_id INTEGER PRIMARY KEY AUTOINCREMENT,
                graph_id INTEGER,
                recommended_method TEXT,
                confidence REAL,
                actual_best_method TEXT,
                is_correct BOOLEAN,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (graph_id) REFERENCES graphs (graph_id)
            )
        ''')
        
        self.conn.commit()

    def calcular_f1_macro(self, G, embedding):
        """Calcula F1-score macro para avaliação de embedding"""
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score
        
        try:
            if isinstance(embedding, dict):
                X = np.array(list(embedding.values()))
            else:
                X = embedding
            
            if len(X) < 3:
                return 0.0
                
            # Clusterização com número ótimo de clusters
            n_clusters = min(5, len(X) // 3)
            if n_clusters < 2:
                return 0.0
                
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X)
            
            if len(np.unique(labels)) < 2:
                return 0.0
                
            return silhouette_score(X, labels)
            
        except Exception as e:
            print(f"Erro no cálculo do F1: {str(e)}")
            return 0.0

    def calculate_all_metrics(self, G):
      """Calcula todas as métricas para todos os métodos de embedding com fallback"""
      resultados = {}
      n_nodes = G.number_of_nodes()
    
      for metodo in self.embedding_methods:
        try:
            # Pula métodos complexos para grafos muito pequenos
            if n_nodes < 10 and metodo in ['hope', 'netmf', 'grarep']:
                resultados[metodo] = {
                    'stress': np.nan,
                    'reconstruction_error': np.nan,
                    'f1_macro': np.nan,
                    'execution_time': np.nan
                }
                continue
                
            start_time = datetime.now()
            embedding = default_embeddings[metodo](G)
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # Se o método falhou, tenta fallback para métodos robustos
            if not embedding:
                if metodo in ['deepwalk', 'node2vec', 'walklets']:
                    # Tenta métodos mais simples
                    embedding = default_embeddings['degree'](G)
                else:
                    embedding = default_embeddings['pagerank'](G)
            
            if not embedding:
                raise ValueError("Todos os métodos de fallback falharam")
            
            # Calcula métricas
            resultados[metodo] = {
                'stress': calcular_stress(G, embedding),
                'reconstruction_error': reconstruction_error(G, embedding),
                'f1_macro': self.calcular_f1_macro(G, embedding),
                'execution_time': execution_time
            }
            
        except Exception as e:
            print(f"Erro no método {metodo}: {str(e)}")
            resultados[metodo] = {
                'stress': np.nan,
                'reconstruction_error': np.nan,
                'f1_macro': np.nan,
                'execution_time': np.nan
            }
    
      return resultados
    def generate_synthetic_graphs(self, n_graphs_per_type=50):
      """Gera grafos sintéticos com tamanhos controlados"""
      graph_types = {
        'erdos_renyi': lambda: nx.erdos_renyi_graph(np.random.randint(30, 100), 0.15),
        'watts_strogatz': lambda: nx.watts_strogatz_graph(np.random.randint(30, 100), 4, 0.3),
        'barabasi_albert': lambda: nx.barabasi_albert_graph(np.random.randint(30, 100), 2),
        'complete': lambda: nx.complete_graph(np.random.randint(20, 40)),
        'star': lambda: nx.star_graph(np.random.randint(20, 40)),
        'wheel': lambda: nx.wheel_graph(np.random.randint(20, 40)),
        'grid': lambda: nx.grid_2d_graph(np.random.randint(6, 10), np.random.randint(6, 10)),
    }
    
      grafos = []
      labels = []
      graph_types_list = []
    
      for graph_type, generator in graph_types.items():
        for _ in range(n_graphs_per_type):
            try:
                G = generator()
                G = nx.convert_node_labels_to_integers(G)
                
                # Garante que o grafo seja conectado e tenha tamanho mínimo
                if (G.number_of_nodes() >= 20 and 
                    G.number_of_edges() >= 10 and 
                    nx.is_connected(G)):
                    grafos.append(G)
                    labels.append(graph_type)
                    graph_types_list.append(graph_type)
                    
            except Exception as e:
                print(f"Erro ao gerar grafo {graph_type}: {str(e)}")
                continue
    
      return grafos, labels, graph_types_list
    
    def fallback_embedding(self, G):
      """Embedding de fallback para quando métodos principais falham"""
      try:
        # Tenta métodos mais robustos primeiro
        for metodo in ['deepwalk', 'node2vec', 'walklets']:
            embedding = default_embeddings[metodo](G)
            if embedding:
                return embedding
        
        # Fallback final: embedding baseado em graus
        degrees = np.array([d for n, d in G.degree()])
        if len(degrees) == 0:
            return {}
        
        degrees_normalized = degrees / np.max(degrees) if np.max(degrees) > 0 else degrees
        embedding = {i: np.array([degrees_normalized[i]], dtype=np.float32) for i in range(len(degrees))}
        return embedding
        
      except Exception:
        return {}

    def save_to_database(self, G, graph_type, metrics_results):
        """Salva grafo e resultados no banco de dados"""
        cursor = self.conn.cursor()
        
        # Salva informações do grafo
        features = extrair_features_grafo(G)
        cursor.execute('''
            INSERT INTO graphs (graph_type, n_nodes, n_edges, density, features)
            VALUES (?, ?, ?, ?, ?)
        ''', (graph_type, G.number_of_nodes(), G.number_of_edges(), 
              nx.density(G), json.dumps(features)))
        
        graph_id = cursor.lastrowid
        
        # Salva resultados dos embeddings
        for metodo, metrics in metrics_results.items():
            if not np.isnan(metrics['f1_macro']):  # Apenas salva resultados válidos
                cursor.execute('''
                    INSERT INTO embedding_results 
                    (graph_id, embedding_method, stress, reconstruction_error, f1_macro, execution_time)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (graph_id, metodo, metrics['stress'], metrics['reconstruction_error'], 
                      metrics['f1_macro'], metrics['execution_time']))
        
        self.conn.commit()
        return graph_id

    def get_best_empirical_method(self, graph_id):
        """Retorna o melhor método empírico baseado nas métricas"""
        cursor = self.conn.cursor()
        
        # Busca o método com maior F1-score (poderia usar combinação de métricas)
        cursor.execute('''
            SELECT embedding_method, f1_macro 
            FROM embedding_results 
            WHERE graph_id = ? 
            ORDER BY f1_macro DESC 
            LIMIT 1
        ''', (graph_id,))
        
        result = cursor.fetchone()
        return result[0] if result else None

    def prepare_training_data(self):
        """Prepara dados de treinamento a partir do banco de dados"""
        cursor = self.conn.cursor()
        
        # Busca todos os grafos com resultados válidos
        cursor.execute('''
            SELECT g.graph_id, g.features, er.embedding_method, er.f1_macro, er.stress, er.reconstruction_error
            FROM graphs g
            JOIN embedding_results er ON g.graph_id = er.graph_id
            WHERE er.f1_macro IS NOT NULL
        ''')
        
        data = cursor.fetchall()
        
        if not data:
            raise ValueError("Nenhum dado válido encontrado no banco de dados")
        
        # Agrupa por grafo
        graph_data = {}
        for row in data:
            graph_id, features_json, method, f1, stress, recon_error = row
            features = json.loads(features_json)
            
            if graph_id not in graph_data:
                graph_data[graph_id] = {
                    'features': list(features.values()),
                    'metrics': {m: {} for m in self.embedding_methods}
                }
            
            graph_data[graph_id]['metrics'][method] = {
                'f1_macro': f1,
                'stress': stress,
                'reconstruction_error': recon_error
            }
        
        # Prepara X e y para treinamento
        X, y = [], []
        
        for graph_id, data in graph_data.items():
            features = data['features']
            metrics = data['metrics']
            
            # Adiciona métricas de todos os métodos como features
            for metodo in self.embedding_methods:
                method_metrics = metrics.get(metodo, {})
                features.extend([
                    method_metrics.get('f1_macro', 0),
                    method_metrics.get('stress', 0),
                    method_metrics.get('reconstruction_error', 0)
                ])
            
            # Encontra o melhor método empírico
            best_method = None
            best_score = -1
            
            for metodo, method_metrics in metrics.items():
                if method_metrics.get('f1_macro', 0) > best_score:
                    best_score = method_metrics['f1_macro']
                    best_method = metodo
            
            if best_method:
                X.append(features)
                y.append(best_method)
        
        return np.array(X), np.array(y)

    def train_model(self):
        """Treina o modelo Random Forest com os parâmetros especificados"""
        X, y = self.prepare_training_data()
        
        # Divide em treino e teste
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=42, stratify=y
        )
        
        # Treina o modelo
        self.model = RandomForestClassifier(**self.rf_params)
        self.model.fit(X_train, y_train)
        
        # Avalia no conjunto de teste
        train_accuracy = self.model.score(X_train, y_train)
        test_accuracy = self.model.score(X_test, y_test)
        
        print(f"Acurácia treino: {train_accuracy:.3f}")
        print(f"Acurácia teste: {test_accuracy:.3f}")
        
        return train_accuracy, test_accuracy

    def predict_best_method(self, G):
        """Prediz o melhor método para um grafo usando o modelo treinado"""
        if self.model is None:
            raise ValueError("Modelo não treinado. Execute train_model() primeiro.")
        
        # Extrai features do grafo
        features = list(extrair_features_grafo(G).values())
        
        # Calcula métricas para todos os métodos (para features adicionais)
        metrics = self.calculate_all_metrics(G)
        
        # Prepara o vetor de features
        X = features.copy()
        for metodo in self.embedding_methods:
            method_metrics = metrics.get(metodo, {})
            X.extend([
                method_metrics.get('f1_macro', 0),
                method_metrics.get('stress', 0),
                method_metrics.get('reconstruction_error', 0)
            ])
        
        # Faz a predição
        X_array = np.array(X).reshape(1, -1)
        prediction = self.model.predict(X_array)[0]
        confidence = np.max(self.model.predict_proba(X_array))
        
        return prediction, confidence, metrics

    def run_experiment_loop(self):
        """Executa o loop completo de experimentos"""
        accuracy_results = []
        
        for run in tqdm(range(self.n_runs), desc="Executando experimentos"):
            try:
                print(f"\n=== Execução {run + 1}/{self.n_runs} ===")
                
                # 1. Gera grafos sintéticos
                print("Gerando grafos sintéticos...")
                grafos, labels, graph_types = self.generate_synthetic_graphs(n_graphs_per_type=20)
                
                # 2. Calcula métricas e salva no banco
                print("Calculando métricas para todos os métodos...")
                for i, G in enumerate(tqdm(grafos, desc="Processando grafos")):
                    metrics = self.calculate_all_metrics(G)
                    self.save_to_database(G, graph_types[i], metrics)
                
                # 3. Treina o modelo
                print("Treinando modelo...")
                train_acc, test_acc = self.train_model()
                
                # 4. Testa as recomendações
                print("Testando recomendações...")
                correct_predictions = 0
                total_predictions = 0
                
                # Usa um subconjunto para teste
                test_indices = np.random.choice(len(grafos), min(20, len(grafos)), replace=False)
                
                for idx in test_indices:
                    G = grafos[idx]
                    graph_type = graph_types[idx]
                    
                    # Prediz o melhor método
                    predicted_method, confidence, metrics = self.predict_best_method(G)
                    
                    # Obtém o melhor método empírico
                    empirical_best = self.get_best_empirical_method(self.get_last_graph_id())
                    
                    if empirical_best:
                        is_correct = (predicted_method == empirical_best)
                        correct_predictions += int(is_correct)
                        total_predictions += 1
                        
                        # Salva a recomendação no banco
                        self.save_recommendation(
                            self.get_last_graph_id(), predicted_method, 
                            confidence, empirical_best, is_correct
                        )
                
                # Calcula acurácia desta execução
                if total_predictions > 0:
                    run_accuracy = correct_predictions / total_predictions
                    accuracy_results.append(run_accuracy)
                    print(f"Acurácia desta execução: {run_accuracy:.3f}")
                
            except Exception as e:
                print(f"Erro na execução {run + 1}: {str(e)}")
                continue
        
        # 5. Analisa resultados finais
        return self.analyze_final_results(accuracy_results)

    def get_last_graph_id(self):
        """Retorna o ID do último grafo inserido"""
        cursor = self.conn.cursor()
        cursor.execute('SELECT MAX(graph_id) FROM graphs')
        return cursor.fetchone()[0]

    def save_recommendation(self, graph_id, recommended_method, confidence, actual_best, is_correct):
        """Salva uma recomendação no banco de dados"""
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT INTO recommendations 
            (graph_id, recommended_method, confidence, actual_best_method, is_correct)
            VALUES (?, ?, ?, ?, ?)
        ''', (graph_id, recommended_method, confidence, actual_best, is_correct))
        self.conn.commit()

    def analyze_final_results(self, accuracy_results):
        """Analisa os resultados finais de todas as execuções"""
        if not accuracy_results:
            return {"error": "Nenhum resultado válido foi coletado"}
        
        df = pd.DataFrame({
            'run': range(1, len(accuracy_results) + 1),
            'accuracy': accuracy_results
        })
        
        stats = {
            'mean_accuracy': np.mean(accuracy_results),
            'std_accuracy': np.std(accuracy_results),
            'min_accuracy': np.min(accuracy_results),
            'max_accuracy': np.max(accuracy_results),
            'median_accuracy': np.median(accuracy_results),
            'n_successful_runs': len(accuracy_results),
            'total_accuracy': np.mean(accuracy_results)
        }
        
        print("\n" + "="*50)
        print("RESULTADOS FINAIS DO EXPERIMENTO")
        print("="*50)
        print(f"Execuções bem-sucedidas: {stats['n_successful_runs']}/{self.n_runs}")
        print(f"Acurácia média: {stats['mean_accuracy']:.3f} ± {stats['std_accuracy']:.3f}")
        print(f"Melhor execução: {stats['max_accuracy']:.3f}")
        print(f"Pior execução: {stats['min_accuracy']:.3f}")
        print(f"Mediana: {stats['median_accuracy']:.3f}")
        print(f"Acurácia total: {stats['total_accuracy']:.3f}")
        
        # Salva resultados detalhados
        df.to_csv("resultados_detalhados.csv", index=False)
        
        # Gera relatório do banco de dados
        self.generate_database_report()
        
        return stats

    def generate_database_report(self):
        """Gera um relatório completo do banco de dados"""
        cursor = self.conn.cursor()
        
        # Estatísticas gerais
        cursor.execute('SELECT COUNT(*) FROM graphs')
        total_graphs = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(*) FROM embedding_results')
        total_results = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(*) FROM recommendations WHERE is_correct = 1')
        correct_recommendations = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(*) FROM recommendations')
        total_recommendations = cursor.fetchone()[0]
        
        accuracy = correct_recommendations / total_recommendations if total_recommendations > 0 else 0
        
        print(f"\nRelatório do Banco de Dados:")
        print(f"Total de grafos: {total_graphs}")
        print(f"Total de resultados de embedding: {total_results}")
        print(f"Total de recomendações: {total_recommendations}")
        print(f"Recomendações corretas: {correct_recommendations}")
        print(f"Acurácia geral: {accuracy:.3f}")
        
        # Métodos mais frequentes como melhor empírico
        cursor.execute('''
            SELECT actual_best_method, COUNT(*) as count
            FROM recommendations
            GROUP BY actual_best_method
            ORDER BY count DESC
        ''')
        
        print("\nMétodos mais frequentes como melhor empírico:")
        for method, count in cursor.fetchall():
            print(f"  {method}: {count} vezes")

    def close(self):
        """Fecha a conexão com o banco de dados"""
        if hasattr(self, 'conn'):
            self.conn.close()

# Exemplo de uso
if __name__ == "__main__":
    # Parâmetros personalizáveis do Random Forest
    custom_rf_params = {
        'n_estimators': 200,
        'max_depth': 15,
        'random_state': 42,
        'min_samples_split': 5,
        'min_samples_leaf': 2,
        'max_features': 'sqrt'
    }
    
    # Cria e executa o experimento
    experiment = EmbeddingPredictionExperiment(
        n_runs=5,  # Número de execuções
        test_size=0.2,
        rf_params=custom_rf_params,
        database_path="embedding_benchmark.db"
    )
    
    try:
        results = experiment.run_experiment_loop()
        
        # Salva resultados em arquivo
        pd.DataFrame([results]).to_csv("resultados_finais.csv", index=False)
        print("\nResultados finais salvos em 'resultados_finais.csv'")
        
    finally:
        experiment.close()


