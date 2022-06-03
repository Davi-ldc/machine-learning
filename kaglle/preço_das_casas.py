import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

with open('data/house_prices.csv') as f:
    dados = pd.read_csv(f)

#tira as colunas desnecessarias
dados = dados.drop(['id', 'date'], axis=1)


#divide os dados em treino e teste
treino, teste = train_test_split(dados, test_size=0.20)

dados_previsores_treinamento = treino.iloc[:, 1:].values
classe_treinamento = treino.iloc[:, 0].values

dados_previsores_teste = teste.iloc[:, 1:].values
classe_teste = treino.iloc[:, 0].values

#transforma os dados em numeros
lb = LabelEncoder()
for i in range(len(dados_previsores_treinamento[0])):
    dados_previsores_treinamento[:, i] = lb.fit_transform(dados_previsores_treinamento[:, i])

for i in range(len(dados_previsores_teste[0])):
    dados_previsores_teste[:, i] = lb.fit_transform(dados_previsores_teste[:, i])
    
#aplica o algoritmo
svm = SVC(C=2.0)

svm.fit(dados_previsores_treinamento, classe_treinamento)

previzoes = svm.predict(dados_previsores_teste)

pontuação = accuracy_score(classe_teste, previzoes)
print(pontuação)