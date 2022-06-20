import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
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
#figura = plt.figure(figsize=(15,15))
#sns.heatmap(data.corr(), annot=True)#quem tem a maior correlação com o preço é a metragem² de casa
#plt.show() #zip code e id são desnecessários (tem corelação negativa)
#annot = True mostra os valores das correlações
#data.corr() mostra a correlação entre as variaveis

data.drop(['zipcode'], axis=1)
variaveis_previsoras = data.iloc[:, 3:].values
classe = data.iloc[:,2].values

#divide a base d dados
variaveis_previsoras_treino, variaveis_previsoras_teste, classe_treino, classe_teste = train_test_split(variaveis_previsoras, classe, test_size=0.3, random_state=0)


#padronização
scaler_previsor = StandardScaler()
scaler_classe = StandardScaler()
variaveis_previsoras_treino = scaler_previsor.fit_transform(variaveis_previsoras_treino)
variaveis_previsoras_teste = scaler_previsor.transform(variaveis_previsoras_teste)
classe_treino = scaler_classe.fit_transform(classe_treino.reshape(-1, 1))
classe_teste = scaler_classe.transform(classe_teste.reshape(-1, 1))
 


#aplicação do algoritmo
svr = SVR(kernel='rbf', C=2)
#kernel = linear, poly, rbf
#kernel linear é reponsavel por tornar os dados linearmente separaveis
#Para criar um hiperplano não linear, usamos as funções RBF e Polinomial.
#C = constante de regularização
svr.fit(variaveis_previsoras_treino, classe_treino.ravel())
previsoes = svr.predict(variaveis_previsoras_teste)

pontuação = svr.score(variaveis_previsoras_teste, classe_teste)
print(pontuação) # 0.7626840846410782
"""
kernel linear(c=2) = 0.6412486671854154
kernel polinoimal(c=1) = 0.6983434514528427
kernel rbf(c=2) = 0.7626840846410782
"""
