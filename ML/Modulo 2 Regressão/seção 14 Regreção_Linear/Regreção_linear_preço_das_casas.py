from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

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

#print(data.isnull().sum())#n tem dados nulos

#grafico correlacao das variaveis
figura = plt.figure(figsize=(15,15))
sns.heatmap(data.corr(), annot=True)#quem tem a maior correlação com o preço é a metragem² de casa
plt.show() #zip code e id são desnecessários (tem corelação negativa)
#annot = True mostra os valores das correlações
#data.corr() mostra a correlação entre as variaveis

data.drop(['zipcode'], axis=1)
variaveis_previsoras = data.iloc[:, 3:].values
classe = data.iloc[:,2].values
#n vamo usar o id, a data


#divide a base d dados
variaveis_previsoras_treino, variaveis_previsoras_teste, classe_treino, classe_teste = train_test_split(variaveis_previsoras, classe, test_size=0.3, random_state=0)


#aplicação do algoritimo
regreção_linear = LinearRegression()
regreção_linear.fit(variaveis_previsoras_treino, classe_treino)#treina

previzões = regreção_linear.predict(variaveis_previsoras_teste)

#mean absolute error
mean_absolute_erro = abs(previzões - classe_teste).mean()
#abs faz com que a diferença seja sempre positiva


#mean squared error
mean_squared_error = (abs(previzões - classe_teste).mean())**2

#root mean squared error
root_mean_squared_error = ((abs(previzões - classe_teste).mean())**2) **0.5


print(regreção_linear.score(variaveis_previsoras_teste, classe_teste)) #0.6877902899299262

#grafico 3d com a linha de regressão, as previzões e os as classes
from mpl_toolkits.mplot3d import Axes3D
figura = plt.figure(figsize=(15,15))
ax = figura.add_subplot(111, projection='3d')
ax.scatter(variaveis_previsoras_treino[:,0], variaveis_previsoras_treino[:,1], classe_treino, color='blue', label='classe verdadeira')
ax.scatter(variaveis_previsoras_teste[:,0], variaveis_previsoras_teste[:,1], previzões, color='red', label='previzões') 
ax.plot(variaveis_previsoras_treino[:,0], variaveis_previsoras_treino[:,1], regreção_linear.predict(variaveis_previsoras_treino), color='green', label='regressão')
ax.set_xlabel('metragem² de casa')
ax.set_ylabel('metragem² do lote')
ax.set_zlabel('preço')
ax.legend()
plt.show()
 
 
#grafico 2d:
figura = plt.figure(figsize=(15,15))
ax = figura.add_subplot(111)
ax.scatter(variaveis_previsoras_treino[:,0], classe_treino, color='blue', label='classe verdadeira')
ax.scatter(variaveis_previsoras_teste[:,0], previzões, color='red', label='previzões')
ax.set_xlabel('metragem² de casa')
ax.set_ylabel('preço')
ax.legend()
plt.show()
 