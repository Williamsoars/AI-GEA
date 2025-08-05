import networkx as nx
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from scipy.stats import pearsonr, spearmanr
from ai_gea.features import extrair_features_grafo
from ai_gea.defaut_embeddings import default_embeddings
from ai_gea.evaluation import calcular_stress, reconstruction_error, calcular_coesao_clusters

# Configuração
NUM_GRAFOS = 100  # Número de grafos sintéticos para teste
TAMANHO_GRAFO = 100  # Número médio de nós
METRICAS_EMBEDDING = {
    "stress": calcular_stress,
    "reconstruction_error": reconstruction_error,
    # Removed cluster_cohesion due to stability issues
}

def validar_embedding(embedding) -> bool:
    """Validação robusta do embedding"""
    if embedding is None:
        return False
        
    if isinstance(embedding, dict):
        if not all(isinstance(k, (int, np.integer)) for k in embedding.keys()):
            return False
        if not all(isinstance(v, np.ndarray) for v in embedding.values()):
            return False
        return True
    elif isinstance(embedding, np.ndarray):
        return True
    return False
def gerar_grafos_variados(n_grafos: int) -> list[tuple[str, nx.Graph]]:
    """Gera grafos com propriedades estruturais diversas."""
    grafos = []
    
    # Grafos aleatórios com densidades variadas
    for densidade in np.linspace(0.05, 0.3, max(3, n_grafos//3)):
        G = nx.erdos_renyi_graph(TAMANHO_GRAFO, densidade)
        G = nx.convert_node_labels_to_integers(G)
        # Remove nós isolados para evitar problemas
        G.remove_nodes_from(list(nx.isolates(G)))
        grafos.append((f"erdos_renyi_d{densidade:.2f}", G))
    
    # Grafos scale-free com graus médios variados
    for m in range(1, max(3, n_grafos//3)):
        G = nx.barabasi_albert_graph(TAMANHO_GRAFO, m)
        G = nx.convert_node_labels_to_integers(G)
        G.remove_nodes_from(list(nx.isolates(G)))
        grafos.append((f"barabasi_albert_m{m}", G))
    
    # Small-world com probabilidades variadas
    for p in np.linspace(0, 1, max(3, n_grafos//3)):
        G = nx.watts_strogatz_graph(TAMANHO_GRAFO, 4, p)
        G = nx.convert_node_labels_to_integers(G)
        G.remove_nodes_from(list(nx.isolates(G)))
        grafos.append((f"watts_strogatz_p{p:.2f}", G))
    
    return grafos

def calcular_correlacoes(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula correlações entre features de grafos e desempenho de embeddings."""
    resultados = []
    features = [col for col in df.columns if col not in 
               {'grafo', 'modelo', 'stress', 'reconstruction_error'}]
    metricas = ['stress', 'reconstruction_error']  # Usando apenas métricas estáveis
    
    for modelo in df['modelo'].unique():
        for feature in features:
            for metrica in metricas:
                # Filtra e limpa os dados
                subset = df[(df['modelo'] == modelo)].copy()
                subset = subset.replace([np.inf, -np.inf], np.nan).dropna(subset=[feature, metrica])
                
                if len(subset) < 2:
                    continue
                
                try:
                    # Verifica se ainda há valores inválidos
                    if subset[feature].isnull().any() or subset[metrica].isnull().any():
                        continue
                        
                    r_pearson, p_pearson = pearsonr(subset[feature], subset[metrica])
                    r_spearman, p_spearman = spearmanr(subset[feature], subset[metrica])
                    
                    resultados.append({
                        'modelo': modelo,
                        'feature': feature,
                        'metrica': metrica,
                        'pearson_r': r_pearson,
                        'pearson_p': p_pearson,
                        'spearman_r': r_spearman,
                        'spearman_p': p_spearman,
                    })
                except Exception as e:
                    print(f"Erro ao calcular correlações para {modelo}-{feature}-{metrica}: {str(e)}")
                    continue
    
    return pd.DataFrame(resultados)

def plot_correlacoes_avancado(df_corr: pd.DataFrame, df: pd.DataFrame):
    """Gera múltiplos tipos de visualizações de correlação."""
    for metrica in df_corr['metrica'].unique():
        # Heatmap de correlações
        plt.figure(figsize=(12, 8))
        heatmap_data = df_corr[df_corr['metrica'] == metrica].pivot(
            index='feature', columns='modelo', values='pearson_r'
        )
        sns.heatmap(heatmap_data, annot=True, cmap='coolwarm', center=0,
                   vmin=-1, vmax=1, fmt=".2f", annot_kws={"size": 8})
        plt.title(f"Correlação de Pearson: Features vs {metrica.replace('_', ' ').title()}")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"correlacao_pearson_{metrica}.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Gráfico de barras comparando correlações
        plt.figure(figsize=(14, 6))
        df_metric = df_corr[df_corr['metrica'] == metrica]
        sns.barplot(data=df_metric, x='feature', y='pearson_r', hue='modelo')
        plt.title(f"Correlação de Pearson por Feature: {metrica.replace('_', ' ').title()}")
        plt.axhline(0, color='black', linestyle='--', linewidth=0.5)
        plt.xticks(rotation=45)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(f"correlacao_barras_{metrica}.png", dpi=300, bbox_inches='tight')
        plt.close()

def benchmark():
    print("🔍 Gerando grafos variados...")
    grafos = gerar_grafos_variados(NUM_GRAFOS)
    
    resultados = []
    for nome_grafo, G in tqdm(grafos, desc="Processando grafos"):
        try:
            G = nx.convert_node_labels_to_integers(G)
            features = extrair_features_grafo(G)
            
            for modelo_nome, modelo_fn in default_embeddings.items():
                try:
                    embedding = modelo_fn(G)
                    
                    # Verificação robusta do embedding
                    if not validar_embedding(embedding):
                        continue
                        
                    if isinstance(embedding, np.ndarray):
                        embedding = {i: embedding[i] for i in range(len(embedding))}
                    
                    # Calcula apenas métricas estáveis
                    try:
                        stress = calcular_stress(G, embedding)
                        recon_err = reconstruction_error(G, embedding)
                        
                        # Verifica se os valores são finitos
                        if not np.isfinite(stress) or not np.isfinite(recon_err):
                            continue
                            
                        resultados.append({
                            'grafo': nome_grafo,
                            'modelo': modelo_nome,
                            **features,
                            'stress': stress,
                            'reconstruction_error': recon_err
                        })
                    except Exception as e:
                        print(f"Erro ao calcular métricas para {modelo_nome} ({nome_grafo}): {str(e)}")
                        continue
                        
                except Exception as e:
                    print(f"Erro no {modelo_nome} ({nome_grafo}): {str(e)}")
                    continue
        except Exception as e:
            print(f"Erro no grafo {nome_grafo}: {str(e)}")
            continue
    
    # Limpeza final dos dados
    df = pd.DataFrame(resultados)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    
    if df.empty:
        raise ValueError("Nenhum dado válido foi gerado. Verifique os logs de erro.")
    
    df.to_csv("resultados_benchmark.csv", index=False)
    
    print("📊 Calculando correlações...")
    try:
        df_corr = calcular_correlacoes(df)
        df_corr.to_csv("correlacoes.csv", index=False)
        
        print("🖼️ Gerando visualizações...")
        plot_correlacoes_avancado(df_corr, df)
    except Exception as e:
        print(f"Erro ao gerar visualizações: {str(e)}")
    
    print("✅ Análise concluída!")

if __name__ == "__main__":
    benchmark()

