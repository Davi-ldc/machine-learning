#mlhr q regreção linear,
#sua linha n precisa ser reta

from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

data = pd.read_csv('data/house_prices.csv')

#grafico correlacao das variaveis
figura = plt.figure(figsize=(15,15))
sns.heatmap(data.corr(), annot=True)#quem tem a maior correlação com o preço é a metragem² de casa
# plt.show() #zip code e id são desnecessários (tem corelação negativa)


variaveis_previsoras = data.iloc[:, 3:19].values
classe = data.iloc[:,2].values


#divide a base d dados
variaveis_previsoras_treino, variaveis_previsoras_teste, classe_treino, classe_teste = train_test_split(variaveis_previsoras, classe, test_size=0.3, random_state=0)


print(variaveis_previsoras_treino.shape, variaveis_previsoras_teste.shape) #(15129, 16) (6484, 16)
poly = PolynomialFeatures(degree=2)
variaveis_previsoras_treinamento_poly = poly.fit_transform(variaveis_previsoras_treino)
variaveis_previsoras_teste_poly = poly.fit_transform(variaveis_previsoras_teste)
print(variaveis_previsoras_treinamento_poly.shape, variaveis_previsoras_teste_poly.shape) #(15129, 153) (6484, 153)

"""
notas:
regressao linear: Y = ax+b
regressao poly: Y = a(x^2)+b

PolynomialFeatures faz com que n tenha como separar os dados com uma linha reta criando novas colunas que são basicamente
as colunas q vc ja tem combinas (tipo se tem 3 colunas abc ele cria: 
aa
ab
ac
ba
bb
bc... #na pratica a conta é (numero d colunas * 2) dividido por 2 pq tipo:
ba e ab tem o msm valor ai n precisa amazenar 2 colunas iguais


regressão linear simples gera uma linha pra classificar os dados linearmente separáveis,
Já regressão polinomial torna os dados n linearmente separados e gera uma linha n linear
"""


#padronzação:
scaler = StandardScaler()
variaveis_previsoras_treinamento_poly = scaler.fit_transform(variaveis_previsoras_treinamento_poly)
variaveis_previsoras_teste_poly = scaler.transform(variaveis_previsoras_teste_poly)
classe_treino = scaler.fit_transform(classe_treino.reshape(-1,1))
classe_teste = scaler.transform(classe_teste.reshape(-1,1))

regreção_polinomial = LinearRegression()
regreção_polinomial.fit(variaveis_previsoras_treinamento_poly, classe_treino)

pontuação = regreção_polinomial.score(variaveis_previsoras_teste_poly, classe_teste)
print(pontuação) #0.8153919693733931

previzões = regreção_polinomial.predict(variaveis_previsoras_teste_poly)