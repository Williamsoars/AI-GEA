

# analysis.py - Advanced embedding analysis and statistical evaluation

import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.stats import shapiro, levene, kruskal
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import scikit_posthocs as sp
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from typing import Dict, List

from ai_gea.defaut_embeddings import (
    deepwalk_embedding, 
    walklets_embedding, 
    hope_embedding,
    node2vec_embedding,
    netmf_embedding,
    grarep_embedding
)
from ai_gea.evaluation import calcular_stress, reconstruction_error

default_embeddings = {
    "deepwalk": deepwalk_embedding,
    "walklets": walklets_embedding,
    "hope": hope_embedding,
    "node2vec": node2vec_embedding,
    "netmf": netmf_embedding,
    "grarep": grarep_embedding
}

def gerar_grafo(tipo: str) -> nx.Graph:
    match tipo:
        case "complete": return nx.complete_graph(30)
        case "cycle": return nx.cycle_graph(30)
        case "path": return nx.path_graph(30)
        case "star": return nx.star_graph(29)
        case "wheel": return nx.wheel_graph(30)
        case "tree": return nx.balanced_tree(2, 4)
        case "barbell": return nx.barbell_graph(15, 5)
        case "grid": return nx.grid_2d_graph(5, 6)
        case _: raise ValueError(f"Grafo '{tipo}' não reconhecido.")

def plot_dendrogram(df: pd.DataFrame) -> None:
    mean_values = df.groupby("Tipo")["Métrica"].mean()
    dist = pdist(mean_values.values.reshape(-1, 1), metric='euclidean')
    linkage_matrix = linkage(dist, method='ward')
    plt.figure(figsize=(10, 5))
    dendrogram(linkage_matrix, labels=mean_values.index.tolist(), leaf_rotation=45)
    plt.title("Dendrograma de Similaridade entre Estruturas de Grafos")
    plt.ylabel("Distância Euclidiana (entre médias)")
    plt.tight_layout()
    plt.show()

def plot_heatmap_posthoc(p_matrix: pd.DataFrame, title: str) -> None:
    plt.figure(figsize=(10, 8))
    sns.heatmap(p_matrix, annot=True, fmt=".3f", cmap="coolwarm_r",
                cbar_kws={'label': 'p-valor ajustado'}, linewidths=.5)
    plt.title(title)
    plt.yticks(rotation=0)
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()

def calcular_f1_clusterizacao(embedding: Dict, n_clusters: int = 2) -> float:
    # Converter para array numpy de forma segura
    if isinstance(embedding, np.ndarray):
        X = embedding
    else:
        X = np.array(list(embedding.values()))
    
    # Verificar dados suficientes
    if len(X) < n_clusters:
        return -1
 
    km = KMeans(n_clusters=n_clusters)
    labels = km.fit_predict(X)
    try:
        return silhouette_score(X, labels)
    except ValueError:
        return -1

def analise_estatistica(df: pd.DataFrame) -> None:
    print("\n=== Análise Estatística ===")
    grupos = [df[df["Tipo"] == tipo]["Métrica"].values for tipo in df["Tipo"].unique()]
    print("\nTeste de normalidade (Shapiro-Wilk):")
    for tipo in df["Tipo"].unique():
        stat, p = shapiro(df[df["Tipo"] == tipo]["Métrica"])
        print(f"{tipo}: p = {p:.4f} {'(normal)' if p > 0.05 else '(não normal)'}")
    stat, p = levene(*grupos)
    print(f"\nLevene (homocedasticidade): p = {p:.4f} {'(variâncias iguais)' if p > 0.05 else '(variâncias diferentes)'}")
    print("\nANOVA:")
    modelo = ols("Métrica ~ C(Tipo)", data=df).fit()
    print(sm.stats.anova_lm(modelo, typ=2))
    print("\nKruskal-Wallis:")
    stat, p = kruskal(*grupos)
    print(f"H = {stat:.4f}, p = {p:.4f} {'(diferença significativa)' if p < 0.05 else '(sem diferença)'}")
    print("\nPost-hoc (Tukey HSD ou Dunn):")
    normalidade = all(shapiro(df[df["Tipo"] == tipo]["Métrica"])[1] > 0.05 for tipo in df["Tipo"].unique())
    if normalidade:
        tukey = pairwise_tukeyhsd(df['Métrica'], df['Tipo'])
        print(tukey)
        tukey_p = pd.DataFrame(np.ones((len(tukey.groupsunique), len(tukey.groupsunique))),
                               columns=tukey.groupsunique, index=tukey.groupsunique)
        for i in range(len(tukey.meandiffs)):
            g1 = tukey.groupsunique[tukey.pairindices[i][0]]
            g2 = tukey.groupsunique[tukey.pairindices[i][1]]
            tukey_p.loc[g1, g2] = tukey.pvalues[i]
            tukey_p.loc[g2, g1] = tukey.pvalues[i]
        plot_heatmap_posthoc(tukey_p, "Heatmap Post-hoc (Tukey HSD)")
    else:
        dunn = sp.posthoc_dunn(df, val_col='Métrica', group_col='Tipo', p_adjust='bonferroni')
        print("\nDunn com correção Bonferroni:")
        print(dunn)
        plot_heatmap_posthoc(dunn, "Heatmap Post-hoc (Dunn-Bonferroni)")
    plot_dendrogram(df)

def avaliar_embeddings(modelo_nome: str, metrica_nome: str = "f1", grafos_tipos: List[str] = None, n_reps: int = 10) -> pd.DataFrame:
    if grafos_tipos is None:
        grafos_tipos = ["complete", "cycle", "path", "star", "wheel", "tree", "barbell", "grid"]
    if modelo_nome not in default_embeddings:
        raise ValueError(f"Modelo {modelo_nome} não encontrado nos embeddings padrão")
    resultados = []
    for tipo in grafos_tipos:
        for _ in range(n_reps):
            G = gerar_grafo(tipo)
            G = nx.convert_node_labels_to_integers(G)
            embedding = default_embeddings[modelo_nome](G)
            if metrica_nome == "stress":
                valor = calcular_stress(G, embedding)
            elif metrica_nome == "f1":
                valor = calcular_f1_clusterizacao(embedding)
            elif metrica_nome == "reconstruction":
                valor = reconstruction_error(G, embedding)
            else:
                raise ValueError("Métrica não reconhecida.")
            resultados.append({"Tipo": tipo, "Métrica": valor})
    df = pd.DataFrame(resultados)
    df.to_csv(f"resultados_{modelo_nome}_{metrica_nome}.csv", index=False)
    plt.figure(figsize=(15, 6))
    sns.boxplot(data=df, x="Tipo", y="Métrica")
    plt.title(f"{metrica_nome.capitalize()} distribution by graph structure in {modelo_nome}")
    plt.xticks(rotation=45)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.show()
    analise_estatistica(df)
    return df

if __name__ == "__main__":
    for modelo in ["deepwalk", "walklets", "hope"]:
        for metrica in ["f1", "stress", "reconstruction"]:
            print(f"\n>>> Rodando: {modelo} - {metrica}")
            avaliar_embeddings(modelo, metrica)

