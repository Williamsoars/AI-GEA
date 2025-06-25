# AI-GEA: Artificial Intelligence for Graph Embedding Analysis

[![Python Version](https://img.shields.io/badge/python-3.7%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Um sistema de recomendação de embeddings para grafos com IA, oferecendo múltiplas funções para análise e recomendação de embeddings.

## Instalação

```bash
pip install AI-GEA
```

## Funcionalidades

- Recomendação automática do melhor método de embedding para um grafo
- Análise comparativa de diferentes métodos de embedding
- Sistema de fila para treinamento assíncrono
- Extração automática de features de grafos
- Plugins para métricas personalizadas
- Logging de recomendações

## Exemplos de Uso

### 1. Recomendação Básica

```python
import networkx as nx
from AI-GEA import Recommender

# Crie um grafo de exemplo
G = nx.karate_club_graph()

# Instancie o recomendador
recommender = EmbeddingRecommender()

# Obtenha recomendações
metodo, scores = recommender.recomendar(G)

print(f"Método recomendado: {metodo}")
print(f"Scores: {scores}")
```

### 2. Treinamento Personalizado

```python
from AI-GEA import Recommender
import networkx as nx

# Grafos de treinamento
grafos = [nx.erdos_renyi_graph(100, 0.1) for _ in range(5)]

# Treine com seus próprios dados
recommender = EmbeddingRecommender()
recommender.treinar(grafos, resultados_metricas)

# Salve o modelo treinado
recommender.salvar_modelo("meu_modelo.pkl")
```

### 3. Usando a Fila de Treinamento

```python
from AI-GEA import Fila_treinamento, treinar_com_fila
import networkx as nx

# Adicione grafos à fila
fila = FilaTreinamento()
G = nx.karate_club_graph()
fila.adicionar(G, {"Node2Vec": 0.95, "DeepWalk": 0.93})

# Processar a fila
treinar_com_fila(modelo_path="modelo_treinado.pkl")
```

### 4. Adicionando Métricas Personalizadas

```python
from ai-gea.plugins import registrar_metrica
import networkx as nx

def minha_metrica(G, embedding):
    return nx.average_clustering(G) * embedding.shape[1]

registrar_metrica("clustering_dim", minha_metrica)

# Agora sua métrica estará disponível em todas as análises
```

### 5. Carregando Grafos de Arquivos

```python
from graph_embedding_recommender.graph_loader import carregar_grafo

G = carregar_grafo("meu_grafo.graphml")
```

## API de Referência

### `EmbeddingRecommender`

Classe principal para recomendações de embeddings.

**Métodos**:
- `recomendar(G, metricas_resultantes=None)`: Recomenda um método de embedding para o grafo G
- `treinar(grafos, resultados)`: Treina o modelo com grafos e resultados conhecidos
- `salvar_modelo(caminho)`: Salva o modelo treinado em um arquivo

### `FilaTreinamento`

Gerencia uma fila de grafos para treinamento assíncrono.

**Métodos**:
- `adicionar(grafo, metricas)`: Adiciona um grafo à fila de treinamento
- `obter_todos()`: Retorna todos os grafos na fila
- `limpar()`: Remove todos os grafos da fila

## Contribuição

Contribuições são bem-vindas! Siga estes passos:

1. Faça um fork do projeto
2. Crie uma branch para sua feature (`git checkout -b feature/AmazingFeature`)
3. Commit suas mudanças (`git commit -m 'Add some AmazingFeature'`)
4. Push para a branch (`git push origin feature/AmazingFeature`)
5. Abra um Pull Request

## Licença

Distribuído sob a licença MIT. Veja `LICENSE` para mais informações.

## Contato

William Silva - Williamkauasoaresdasilva@gmail.com

Link do Projeto: [https://github.com/Williamsoars/AI-GEA](https://github.com/Williamsoars/AI-GEA)
