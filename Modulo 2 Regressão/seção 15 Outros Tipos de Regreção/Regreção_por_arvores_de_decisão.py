import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error

with open('data/house_prices.csv', 'r') as f:
    data = pd.read_csv(f)
"""
#price(preço)
#bedrooms(quartos)
#bathrooms(banheiros)
#sqft_living(metragem² de casa)
#sqft_lot(metragem² do lote)
#floors(andares)
#waterfront(frente d'água)
#view(vista)
#condition(condição)
#grade(grau)
#sqft_above(metragem² acima)
#sqft_basement(metragem² do fundo)
#yr_built(ano de construção)
#yr_renovated(ano de renovação)
#lat(latitude)
#long(longitude)
#sqft_living15(metragem² de casa no ultimo ano)
#sqft_lot15(metragem² do lote no ultimo ano)
"""


#grafico correlacao das variaveis
figura = plt.figure(figsize=(15,15))
sns.heatmap(data.corr(), annot=True)#quem tem a maior correlação com o preço é a metragem² de casa
plt.show() #zip code e id são desnecessários (tem corelação negativa)
#annot = True mostra os valores das correlações
#data.corr() mostra a correlação entre as variaveis

data.drop(['zipcode'], axis=1)
variaveis_previsoras = data.iloc[:, 3:].values
classe = data.iloc[:,2].values

#divide a base d dados
variaveis_previsoras_treino, variaveis_previsoras_teste, classe_treino, classe_teste = train_test_split(variaveis_previsoras, classe, test_size=0.3, random_state=0)


#padronização
scaler = StandardScaler()
scaler.fit(X=variaveis_previsoras_treino, y=classe_treino)
variaveis_previsoras_treino = scaler.transform(variaveis_previsoras_treino)
variaveis_previsoras_teste = scaler.transform(variaveis_previsoras_teste)
classe_treino = scaler.fit_transform(classe_treino.reshape(-1, 1))
classe_teste = scaler.fit_transform(classe_teste.reshape(-1, 1))
 


#aplicação do algoritmo
arvore = DecisionTreeRegressor(random_state=0)
arvore.fit(variaveis_previsoras_treino, classe_treino)#treina

previzões = arvore.predict(variaveis_previsoras_teste)
erro = mean_absolute_error(classe_teste, previzões)
print(erro)#0.2684743760903499

pontuação = arvore.score(variaveis_previsoras_teste, classe_teste)
print(pontuação)#0.7582442245931977


#grafico pontuação algoritimo 
