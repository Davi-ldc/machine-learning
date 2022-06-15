#cria novos dados
#Não tem uma classe
#o algoritmo de aprende as relações entre os dados

"""
resumo k-means

tenta descobrir o centro de um grupo de dados atravez da distancia euclidiana
DE(x,y) = raiz_quadrada(somatorio(xi-yi)^2)
os contros começam aleatórios e dps eles são atualizados
"""

"""
k-means++
msm coisa que k-means, mas ao inves de escolher os centros aleatoriamente,
ele escolhe os centros de forma mais eficiente (usando a distancia entre els)
"""

import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

variaveis_previsoras_aleatorias, classes_aleatórias = make_blobs(n_samples=200, centers=5, cluster_std=1.0)

grafico_dados = px.scatter(x=variaveis_previsoras_aleatorias[:, 0], y=variaveis_previsoras_aleatorias[:, 1])
grafico_dados.show()



kmeans = KMeans(n_clusters=5)
kmeans.fit(variaveis_previsoras_aleatorias)

previzões = kmeans.predict(variaveis_previsoras_aleatorias)

grafico_kmeans = px.scatter(x=variaveis_previsoras_aleatorias[:, 0], y=variaveis_previsoras_aleatorias[:, 1], color=previzões)
grafico_kmeans2 = px.scatter(x=kmeans.cluster_centers_[:, 0], y=kmeans.cluster_centers_[:, 1], size=[5, 5, 5, 5, 5])
grafico_kmeans3 = go.Figure(data=grafico_kmeans2.data + grafico_kmeans.data)
grafico_kmeans3.show()