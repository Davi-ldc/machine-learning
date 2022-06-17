import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

data = pd.read_csv('data/creditcard.csv')

#junta as colunas
data['BILL_AMT_TOTAL'] = data['BILL_AMT1'] + data['BILL_AMT2'] + data['BILL_AMT3'] + data['BILL_AMT4'] + data['BILL_AMT5'] + data['BILL_AMT6']



print(len(data.columns))#25 colunas

variaveis_previsoras = data.iloc[:, [1,2,3,4,5,25]].values

scaler = StandardScaler()
variaveis_previsoras = scaler.fit_transform(variaveis_previsoras)

numeros = []
for c in range(1, 11): #como eu n sei em quantos grupo dividir a base eu vou testar varios valores
    kmeans = KMeans(n_clusters=c, random_state=1)
    kmeans.fit(variaveis_previsoras)
    numeros.append(kmeans.inertia_)#quanto menor o valor, melhor

grafico_parametros = px.line(x=range(1, 11), y=numeros)
grafico_parametros.show()

kmeans_final = KMeans(n_clusters=3, random_state=1)
previsoes = kmeans_final.fit_predict(variaveis_previsoras)# treina e retorna as previsoes

#então, so da pra frz graficom com x e y, com tds as colunas da variavel previsoras n da
#ai pra resolver isso eu vou "juntar os atributos"

from sklearn.decomposition import PCA #Principal Component Analysis
Principal_Component_Analysis = PCA(n_components=2)
#n_components = numeros de atributos que ele vai retornar
variaveis_previsoras_q_da_pra_frz_grafico = Principal_Component_Analysis.fit_transform(variaveis_previsoras)

grafico = px.scatter(data, x=variaveis_previsoras_q_da_pra_frz_grafico[:, 0], y=variaveis_previsoras_q_da_pra_frz_grafico[:, 1], color=previsoes)
grafico.show()
