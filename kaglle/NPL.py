#NÃO FUNCIONA
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer

#carrega os dados
with open('data/emotions_train.txt') as f:
    treinamento = pd.read_csv(f, sep = ';', names=["emotion", "sentence"])

with open('data/emotions_test.txt') as f:
    teste = pd.read_csv(f, sep = ';', names=["emotion", "sentence"])
    
dados = treinamento.append(teste)

cv = CountVectorizer()
vetor = cv.fit(dados['sentence'])


#divide os dados

dados_previsores_treinamento = treinamento.iloc[:, 0].values
classes_treinamento = treinamento.iloc[:, 1].values

dados_previsores_teste = teste.iloc[:, 0].values
classes_teste = teste.iloc[:, 1].values

dados_previsores_treinamento = vetor.transform(dados_previsores_treinamento)
dados_previsores_teste = vetor.transform(dados_previsores_teste)

def emotions(string,vector,model):
     vectorized = vector.transform([string])
     pred = model.predict(vectorized)
     return pred

#aplica a rede neural
from sklearn.neural_network import MLPClassifier

rede_neural = MLPClassifier(verbose=True, max_iter=1000, tol=0.00001, random_state=0)

#treina a rede neural
rede_neural.fit(dados_previsores_treinamento, classes_treinamento)


#testa a rede neural
previsoes = rede_neural.predict(dados_previsores_teste)

#pontuação
pontuação = accuracy_score(classes_teste, previsoes)
print(pontuação)

from yellowbrick.classifier import ClassificationReport

cm = ClassificationReport(floresMLPClassifier)
cm.fit(dados_previsores_treinamento, classes_treinamento)
cm.score(dados_previsores_teste, classes_teste)
cm.poof()