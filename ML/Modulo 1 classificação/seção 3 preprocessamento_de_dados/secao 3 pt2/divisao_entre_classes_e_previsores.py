"""O OBJETIVO É SABER SE ALGUEM GANHA MAIS OU MENOS QUE 50K POR ANO"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

"""Pra frz isso eu vo usar aprendizagem supervisionada (funciona assim:
eu dividi as colunas da base de dados em variaveis previsores e classes ai usando as variaveis previsores o algoritimo
vai tentar descobrir a classe se ele descobrir ele ganha um ponto e se ele erar ele perde um ponto)
e duas base de dados uma pra teste e outra pra treinamento
 """



dados_treinamenro = pd.read_csv('secao 3 pt2/census.csv')
print(len(dados_treinamenro.columns)) # 14 itens

#DIVISÃO ENTRE VARIAVEIS_PREVISORAS E CLASSES
variaveis_previsoras_treinamento = dados_treinamenro.iloc[:, 0:14].values #dados_treinamenro que serão usados para descubrir se 
#a renda anuaria de alguem é maior ou menor que 50k
classes_teste = dados_treinamenro.iloc[:, 14].values #o algoritimo vai retornar algum intem desse coluna( que so tem 2 itens)





#LABEL ENCODER
print(variaveis_previsoras_treinamento[1]) # [50 ' Self-emp-not-inc' 83311 ' Bachelors' 13 ' Married-civ-spouse'
# ' Exec-managerial' ' Husband' ' White' ' Male' 0 0 13 ' United-States']
print(np.unique(variaveis_previsoras_treinamento[:, 1], return_counts=True))
#vou transformar os dados unicos das variaveis_promissoras em numeros

# label = LabelEncoder()
# # transforma cada um dos atributos em numeros (pq se for str n da pra rede neural frz os cauculos)
# for c in range(1, 14):
#     if c == 2 or c == 10 or c == 11 or c == 12: # colunas que ja sao numericas
#         continue
#     variaveis_previsoras_treinamento[:, c] = label.fit_transform(variaveis_previsoras_treinamento[:, c])
    

# print(variaveis_previsoras_treinamento[0])


#OneHotEncoder
"""por mais que com o LabelEncoder de pra rede neural frz os calculos, alguns dados vão
receber valores maiores que outros e isso pode gerar problemas com a rede neural,
(em cauculos de multiplicção valores maiores teram resultados maiores e peso maior ou seja
VALORES QUE RECEBERAM VALORES ALTOS SERÃO MAIS IMPORTANTES (PRA REDE NEURAL) DO QUE 
VALORES BAIXOS)"""

# ai pra resolver isso podemos usar essa linha:
one_hot_encoder = ColumnTransformer(transformers=[("OneHot", OneHotEncoder(), [1, 3, 5, 6, 7, 8, 9, 13])], remainder="passthrough")
#que transforma os valores em 010101001010 
variaveis_previsoras_treinamento = one_hot_encoder.fit_transform(variaveis_previsoras_treinamento).toarray()

print(variaveis_previsoras_treinamento)



#PADRONIZAÇÃO

scaler = StandardScaler()

variaveis_previsoras_treinamento = scaler.fit_transform(variaveis_previsoras_treinamento)

print(variaveis_previsoras_treinamento[0])
print(variaveis_previsoras_treinamento.shape)