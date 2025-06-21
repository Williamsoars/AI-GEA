# Este módulo permite que usuários registrem métricas personalizadas
custom_metrics = {}

def registrar_metrica(nome, funcao):
    custom_metrics[nome] = funcao

def calcular_metricas_personalizadas(G, embedding):
    resultados = {}
    for nome, funcao in custom_metrics.items():
        try:
            resultados[nome] = funcao(G, embedding)
        except Exception as e:
            resultados[nome] = None
    return resultados
