import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

dados= pd.read_csv('secao 3 pt2/dados_teste.csv')


#acha o index de valores falsos ou nulos
indice_idades_falsas = dados[dados['age'] < 0].index
indice_idades_faltando = dados.loc[pd.isnull(dados['age'])].index

#remove eles da base d dados
dados = dados.drop(indice_idades_falsas)
dados = dados.drop(indice_idades_faltando)

variaveis_previsoras = dados.iloc[:, 1:4].values
#seleciona linas e colunas fala que queremos todas as linhas seleciona as 3 ultimas colunas (ele não vai ate o 4)
#pois não queremos q o id do usariario seja levado em consideração pelo algoritimo.
classe = dados.iloc[:, 4].values
padrao = StandardScaler()
variaveis_previsoras = padrao.fit_transform(variaveis_previsoras)

variaveis_previsoras_treinamento, variaveis_previsoras_teste, classe_treinamento, classe_teste = train_test_split(variaveis_previsoras, classe, test_size=0.25, random_state=0)

with open('teste.pkl', 'wb') as f:
    pickle.dump([variaveis_previsoras_treinamento, classe_teste, variaveis_previsoras_teste, classe_teste], f)
    