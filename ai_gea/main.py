import networkx as nx
import pandas as pd
from graph_loader import load_cora_graph
from features import extrair_features_grafo
from defaut_embeddings import deepwalk, walklets, hope
from evaluation import evaluate_embeddings, calcular_stress, reconstruction_error
from analysis import analise_estatistica
import seaborn as sns
import matplotlib.pyplot as plt

# Modelos de embedding disponíveis
modelos = {
    "deepwalk": deepwalk,
    "walklets": walklets,
    "hope": hope
}

# Métricas a avaliar
metricas = {
    "stress": calcular_stress,
    "reconstruction": reconstruction_error,
    "f1": lambda G, emb: evaluate_embeddings(G, emb)["f1_macro"]
}

# 1. Carregar grafo Cora
G = load_cora_graph()

# 2. Extrair características estruturais
print("Extraindo características estruturais do grafo...")
features = extrair_features_grafo(G)
print("Features:", features)

# 3. Avaliar embeddings com diferentes métodos e métricas
resultados = []

for modelo_nome, funcao_embedding in modelos.items():
    print(f"\n--- Rodando modelo: {modelo_nome} ---")
    emb = funcao_embedding(G)

    for metrica_nome, func_metrica in metricas.items():
        print(f"> Avaliando métrica: {metrica_nome}")
        valor = func_metrica(G, emb)

        resultado = {
            "Modelo": modelo_nome,
            "Métrica": metrica_nome,
            "Valor": valor
        }
        resultado.update(features)  # adiciona as features ao dicionário
        resultados.append(resultado)

# 4. Criar DataFrame geral
df = pd.DataFrame(resultados)
df.to_csv("resultados_experimento.csv", index=False)
print("\nResultados salvos em 'resultados_experimento.csv'")

# 5. Visualização
sns.set(style="whitegrid")
for metrica in df["Métrica"].unique():
    df_plot = df[df["Métrica"] == metrica]
    plt.figure(figsize=(10, 5))
    sns.violinplot(data=df_plot, x="Modelo", y="Valor")
    plt.title(f"{metrica.upper()} por Modelo")
    plt.tight_layout()
    plt.savefig(f"violin_{metrica}.png")
    plt.show()

# 6. Análise estatística cruzando modelo × valor da métrica
for metrica in df["Métrica"].unique():
    df_sub = df[df["Métrica"] == metrica][["Modelo", "Valor"]].rename(columns={"Modelo": "Tipo", "Valor": "Métrica"})
    print(f"\nAnalisando estatisticamente a métrica: {metrica}")
    analise_estatistica(df_sub)


