"""O OBJETIVO É SABER SE ALGUEM GANHA MAIS OU MENOS QUE 50K POR ANO"""
import pandas as pd
import numpy as np
import sklearn
import matplotlib.pyplot as plt

dados = pd.read_csv('secao 3 pt2/census.csv')
#age(idade)
#workclass(classe de trabalho)
#final-weight (pontuação das pessoas)
#education (educação)
#education-num (numero de anos que a pessoa estudou)
#marital-status (estado civil)
#occupation (ocupação)
#relationship (relação)
#race (raça) 
#sex(sexo)
#capital-gain(quanto alguem ganhou  (em dinheiro))
#capital-loos (quanto alguem ganhou (em dinheiro))
#hour-per-week (quantas horas alguem trabalhou em 1 semana)
#native-country (pais nativo)
#14 colunas


#income (quanto a pessoa ganha por ano)(pode ser <=50K ou >50K)

#descrição dos dados
print(dados.describe())

print(dados.isnull().sum())# n tem nenhum dado nulo

print(np.unique(dados['income'], return_counts = True)) #(array([' <=50K', ' >50K'], dtype=object), array([24720,  7841], dtype=int64))
# grafico com varias cores
plt.hist(dados['income'])
plt.show()
