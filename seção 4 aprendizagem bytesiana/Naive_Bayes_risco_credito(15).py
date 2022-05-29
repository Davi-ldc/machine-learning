import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
dados = pd.read_csv('seção 4 aprendizagem bytesiana/risco_credito.csv')

#PRE PROCESSAMENTO:
variaveis_previsoras = dados.iloc[:, 0:4].values
classe = dados.iloc[:, 4].values

label = LabelEncoder()

for c in range(0, 4):
    variaveis_previsoras[:, c] = label.fit_transform(variaveis_previsoras[:, c])
    
# with open('risco_credito.pkl', 'wb') as f:
#     pickle.dump([variaveis_previsoras, classe], f)
    

naive = GaussianNB()
# objeto

#treina o algoritimo
naive.fit(variaveis_previsoras, classe)
#faz toda a matematica (o primeiro parametro sempre é as variaveis previsoras e o segundo é a classe)

#hitória boa(0), divida alta(0), garantia nenhuma(1), renda > 35(2)

#previsao = naive.predict(['boa', 'alta', 'nenhuma' '>35']) #erro pq n da pra frz cauculo com numero
previsao = naive.predict([[0,0,1,2], [2,0,0,0]]) # tem q tar dentro d 2 listas
#retorna a classe

print(previsao)
print(naive.classes_) # classes possiveis
print(naive.class_count_) # quantos registros de cada classe
print(naive.class_prior_) # porcentagem que cada registro representa da base de dados