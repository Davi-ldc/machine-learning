import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

with open('data/paga_ou_n_paga.pkl', 'rb') as file:
    #     arquivo      ler
    dados = variaveis_previsoras_treinamento, classe_treinamento, variaveis_previsoras_teste, classe_teste = pickle.load(file)
#defaut 0 é qnd o cliente pagou o emprestimo e 1 é qnd ele n pagou
    
naive = GaussianNB()
#treino

naive.fit(variaveis_previsoras_treinamento, classe_treinamento) #treino

previsão = naive.predict(variaveis_previsoras_teste) # teste

#vendo se ele acertou ou n


print(accuracy_score(classe_teste, previsão)) # acertou 93 porcento das perguntas 
print(classification_report(classe_teste, previsão)) # detalhes sobre o algoritomo

from yellowbrick.classifier import ConfusionMatrix # muito mlhr q sklearn


obj = ConfusionMatrix(naive)
obj.fit(variaveis_previsoras_treinamento, classe_treinamento) # treina

print(obj.score(variaveis_previsoras_teste, classe_teste))
#vendo a matriz de confusão
obj.poof()