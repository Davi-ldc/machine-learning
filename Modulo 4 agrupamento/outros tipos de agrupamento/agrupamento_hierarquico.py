#gera uma estrutura no formato de arvores que indica o numero de grupo
#no inicio cada registro equivale a um grupo
#usa a distancia euclidiana
#conforme o treinamento ele vai juntando os grupos que estão proximos

import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
from scipy.cluster import hierarchy
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering


data = pd.read_csv('data/creditcard.csv')

#junta as colunas
data['BILL_AMT_TOTAL'] = data['BILL_AMT1'] + data['BILL_AMT2'] + data['BILL_AMT3'] + data['BILL_AMT4'] + data['BILL_AMT5'] + data['BILL_AMT6']



print(len(data.columns))#25 colunas

variaveis_previsoras = data.iloc[:, [1,2,3,4,5,25]].values

scaler = StandardScaler()
variaveis_previsoras = scaler.fit_transform(variaveis_previsoras)

print(variaveis_previsoras)

# dendrograma = hierarchy.dendrogram(hierarchy.linkage(variaveis_previsoras, method='ward'))

Agrupamento_hierarquico = AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='ward')
previsões = Agrupamento_hierarquico.fit_predict(variaveis_previsoras)

grafico_agrupamento_hierarquico = px.scatter(x = variaveis_previsoras[:,0], y = variaveis_previsoras[:,1], color = previsões)
grafico_agrupamento_hierarquico.show()