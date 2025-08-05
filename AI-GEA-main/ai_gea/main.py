import networkx as nx
import pandas as pd
from ai_gea.graph_loader import carregar_grafo
from ai_gea.features import extrair_features_grafo
from ai_gea.defaut_embeddings import default_embeddings
from ai_gea.evaluation import calcular_stress, reconstruction_error
from ai_gea.analysis import analise_estatistica
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

# --- Configuração do Benchmark ---
GRAFOS_SINTETICOS = {
    "grafo_aleatorio": nx.erdos_renyi_graph(100, 0.1),
    "grafo_scale_free": nx.barabasi_albert_graph(100, 2),
    "grafo_small_world": nx.watts_strogatz_graph(100, 4, 0.3),
    "grafo_malha": nx.grid_2d_graph(10, 10),
    "grafo_completo": nx.complete_graph(50),
}

# Métricas a avaliar
METRICAS = {
    "stress": calcular_stress,
    "reconstruction": reconstruction_error,
}

def gerar_relatorio(resultados: pd.DataFrame) -> None:
    """Gera visualizações e análises estatísticas."""
    sns.set(style="whitegrid")
    
    # Boxplot por métrica
    for metrica in resultados["Métrica"].unique():
        plt.figure(figsize=(10, 5))
        sns.boxplot(data=resultados[resultados["Métrica"] == metrica], 
                    x="Modelo", y="Valor")
        plt.title(f"Distribuição de {metrica.upper()} por Modelo")
        plt.tight_layout()
        plt.savefig(f"benchmark_{metrica}.png")
        plt.close()

    # Análise estatística
    for metrica in resultados["Métrica"].unique():
        df_sub = resultados[resultados["Métrica"] == metrica]
        print(f"\n=== Análise Estatística para {metrica.upper()} ===")
        analise_estatistica(df_sub[["Modelo", "Valor"]].rename(columns={"Modelo": "Tipo"}))

def benchmark() -> pd.DataFrame:
    """Executa o benchmark em todos os grafos e embeddings."""
    resultados = []
    
    for nome_grafo, G in tqdm(GRAFOS_SINTETICOS.items(), desc="Processando grafos"):
        # Converter labels para inteiros (necessário para alguns embeddings)
        G = nx.convert_node_labels_to_integers(G)
        
        # Extrair features estruturais
        features = extrair_features_grafo(G)
        
        for modelo_nome, modelo_fn in default_embeddings.items():
            try:
                # Gerar embedding
                embedding = modelo_fn(G)
                
                # Avaliar métricas
                for metrica_nome, metrica_fn in METRICAS.items():
                    valor = metrica_fn(G, embedding)
                    
                    resultados.append({
                        "Grafo": nome_grafo,
                        "Modelo": modelo_nome,
                        "Métrica": metrica_nome,
                        "Valor": valor,
                        **features  # Adiciona features estruturais
                    })
                    
            except Exception as e:
                print(f"Erro no modelo {modelo_nome} (grafo {nome_grafo}): {str(e)}")
    
    return pd.DataFrame(resultados)

if __name__ == "__main__":
    print("🚀 Iniciando benchmark...")
    df_resultados = benchmark()
    df_resultados.to_csv("resultados_benchmark.csv", index=False)
    print("✅ Resultados salvos em 'resultados_benchmark.csv'")
    
    gerar_relatorio(df_resultados)
    print("📊 Relatórios gerados!")


