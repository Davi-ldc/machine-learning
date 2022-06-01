import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer

#carrega os dados
with open('data/emotions_train.txt') as f:
    treinamento = pd.read_csv(f, sep = ';', names=["emotion", "sentence"])

with open('data/emotions_test.txt') as f:
    teste = pd.read_csv(f, sep = ';', names=["emotion", "sentence"])

#divide os dados

dados_previsores_treinamento = treinamento.iloc[:, 0].values
classes_treinamento = treinamento.iloc[:, 1].values

dados_previsores_teste = teste.iloc[:, 0].values
classes_teste = teste.iloc[:, 1].values

#substitue cada palavra por um numero, e mostra qual numero cada palavra tem
cv = CountVectorizer()
dados_previsores_treinamento = cv.fit_transform(dados_previsores_treinamento)
classes_treinamento = cv.fit_transform(classes_treinamento)

dados_previsores_teste = cv.fit_transform(dados_previsores_teste)
classes_teste = cv.fit_transform(classes_teste)

def prever(modelo, str):
    #transforma uma str em um numero
    global cv
    #preve o numero
    numero_que_a_str_recebeu = None
    return modelo.predict(numero_que_a_str_recebeu)


#aplica a rede neural
from sklearn.neural_network import MLPClassifier

neural_network = MLPClassifier(verbose=True, max_iter=1000, tol=0.00001, solver='adam', hidden_layer_sizes=(100, 100, 100), random_state=1)

#treina a rede neural
neural_network.fit(dados_previsores_treinamento, classes_treinamento)

#testa a rede neural
previsoes = neural_network.predict(dados_previsores_teste)

#pontuação
pontuação = accuracy_score(classes_teste, previsoes)
print(pontuação)
