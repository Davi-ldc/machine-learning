import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
"""vou usar aprendizagem supervisionada.(funciona assim:
eu vo separar as bases de dados em classes e previsores, o algorito vai usar os dados previsores para
tentar descobrir a classe (que ele não sabe) se ele descobrir a classe ganha um ponto de errar perde um ponto [EX:
(dados previsores: branco, brasilia 23 anos não tem filho e tem um bom histórico (ai o algoritimo vai tentar descobrir se essa pessoa
vai ou n pagar emprestimo sendo que eu sei se q a pessoa pagou ou não (no caso ela pagou) 
se ele falar q ele pagou ganha um ponto se falar q n pagou perde um ponto]))"""

dados= pd.read_csv('data/census.csv')

variaveis_previsoras = dados.iloc[:, 0:14].values #dados que serão usados para descubrir se 
#a renda anuaria de alguem é maior ou menor que 50k
classes = dados.iloc[:, 14].values #o algoritimo vai retornar algum intem desse coluna( que so tem 2 itens)


one_hot_encoder = ColumnTransformer(transformers=[("OneHot", OneHotEncoder(), [1, 3, 5, 6, 7, 8, 9, 13])], remainder="passthrough")
#que transforma os valores em 010101001010 
variaveis_previsoras = one_hot_encoder.fit_transform(variaveis_previsoras).toarray()


#PADRONIZAÇÃO
scaler = StandardScaler()

variaveis_previsoras = scaler.fit_transform(variaveis_previsoras)
"""ta mas cm q sei se o algoritimo sabe oq ta fazendo ou se ele so memorisou quem pagou q qm n pagou o emprestimo?
pra saber se ele sabe oq ta fazendo eu vou dividir a base de dados em 2 uma pra teste (com perguntas pra dps q ele aprender)
e uma pra treinamento (com as perguntas pra ele aprender) quando ele passar no teste eu sei q ele aprendeu"""
variaveis_previsoras_treinamento, variaveis_previsoras_teste, classes_treinamento, classes_teste = train_test_split(variaveis_previsoras, classes, test_size=0.15, random_state=0)
#                                                                                                                                         tamanho da base d test    faz com que os dados sejam aleatorios                     
         
print(variaveis_previsoras_treinamento.shape) # (27676, 108)
print(variaveis_previsoras_teste.shape) # (4885, 108)
# repara q a de test é menor q a de treinamento

#salva as bases d dados
with open('treinamento.pkl', 'wb') as f:
    pickle.dump([variaveis_previsoras_treinamento, classes_teste, variaveis_previsoras_teste, classes_teste], f)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
"""Se treinados por muito tempo, os algoritmos têm a tendência de se adequar demais aos dados de treinamento, fenômeno que chamamos de 
overfitting. É como se, ao invés de aprenderem algo sobre a característica dos dados que seja útil para a predição, eles apenas os 
decorassem. Nesta situação, eles perdem desempenho quando aplicados fora do treinamento, ou seja, tentando fazer predições em novos
dados, que é o objetivo final do treinamento do algoritmo. Para evitar isso, nós buscamos acompanhar seu desempenho com um dataset 
separado. Se a performance neste dataset diminuir, isto indica que o algoritmo está em fase de overfitting, e nós interrompemos o 
treinamento neste ponto."""