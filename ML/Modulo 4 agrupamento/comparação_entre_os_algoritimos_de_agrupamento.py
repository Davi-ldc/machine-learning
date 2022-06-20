from sklearn.datasets import make_moons
import plotly.graph_objects as go
import plotly.express as px
from sklearn.cluster import DBSCAN, KMeans

variaveis_previsoras, classes = make_moons(n_samples=1500)
grafico_dados_muito_juntos = px.scatter(x=variaveis_previsoras[:, 0], y=variaveis_previsoras[:, 1], color=classes)
grafico_dados_muito_juntos.show()


variaveis_previsoras, classes = make_moons(n_samples=1500, noise=0.08, random_state=1)
#noise faz com que os dados fiquem mais dispersos 

grafico_dados_dispersos = px.scatter(x=variaveis_previsoras[:, 0], y=variaveis_previsoras[:,1], color=classes)
grafico_dados_dispersos.show()

kmeans = KMeans(n_clusters=2)
previsoes = kmeans.fit_predict(variaveis_previsoras)

grafico_kmeans = px.scatter(x=variaveis_previsoras[:, 0], y=variaveis_previsoras[:, 1], color=previsoes)
grafico_kmeans.show()

#n vou testar o agrupamento hierarquico pq trava meu pc 

dbscan = DBSCAN(eps=0.135371, min_samples=5)#se o raio for muito grande ele só vai criar um grupo devido a posição dos dados
previsoes = dbscan.fit_predict(variaveis_previsoras)

grafico_dbscan = px.scatter(x=variaveis_previsoras[:, 0], y=variaveis_previsoras[:, 1], color=previsoes)
grafico_dbscan.show()

