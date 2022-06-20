import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

with open('data/census.pkl', 'rb') as file:
    #     arquivo      ler (rb = ler wb = salvar)
    dados = variaveis_previsoras_treinamento, classe_treinamento, variaveis_previsoras_teste, classe_teste = pickle.load(file)
    #carega a base de dados ja pre processada

naive = GaussianNB()

naive.fit(variaveis_previsoras_treinamento, classe_treinamento) # trina

previsão = naive.predict(variaveis_previsoras_teste)
#faz a previsão dos dados d teste sem saber a classa (dps pra saber a porcentagem d acerto do algoritomo
#agnt compara as previsões com as classes de teste

detalhes = classification_report(classe_teste, previsão) # detalhes sobre o algoritomo (faz a comparação entre as classes de teste e as previsões)

print(detalhes)

from yellowbrick.classifier import ConfusionMatrix # muito mlhr q sklearn

obj = ConfusionMatrix(naive)

obj.fit(variaveis_previsoras_treinamento, classe_treinamento) # treina

porcentagem_de_acerto = obj.score(variaveis_previsoras_teste, classe_teste)

print(porcentagem_de_acerto)
#obs sem a linha da porcentagem de acerto o grafico buga
obj.poof() #grafico com os dados do algorito (qnt ele erro e qnt ele acertou)