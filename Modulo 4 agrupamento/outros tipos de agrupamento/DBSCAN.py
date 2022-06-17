#não é necessário especificar o número de grupos
#mais rapido e mlhr q o k-means



from sklearn.cluster import DBSCAN
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

data = pd.read_csv('data/creditcard.csv')

#junta as colunas
data['BILL_AMT_TOTAL'] = data['BILL_AMT1'] + data['BILL_AMT2'] + data['BILL_AMT3'] + data['BILL_AMT4'] + data['BILL_AMT5'] + data['BILL_AMT6']



print(len(data.columns))#25 colunas

variaveis_previsoras = data.iloc[:, [1,2,3,4,5,25]].values

scaler = StandardScaler()
variaveis_previsoras = scaler.fit_transform(variaveis_previsoras)

dbscan = DBSCAN(eps=0.3, min_samples=5)
#esp é o tamnho do raio de vizinhança
#min_samples é o número mínimo de vizinhos pra ser considerado um grupo

previsoes = dbscan.fit_predict(variaveis_previsoras)


#então, so da pra frz graficom com x e y, com tds as colunas da variavel previsoras n da
#ai pra resolver isso eu vou "juntar os atributos"
Principal_Component_Analysis = PCA(n_components=2)
#n_components = numeros de atributos que ele vai retornar
variaveis_previsoras_q_da_pra_frz_grafico = Principal_Component_Analysis.fit_transform(variaveis_previsoras)

grafico = px.scatter(data, x=variaveis_previsoras_q_da_pra_frz_grafico[:, 0], y=variaveis_previsoras_q_da_pra_frz_grafico[:, 1], color=previsoes)
grafico.show()