import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

with open('data/census.pkl', 'rb') as f:
    dados_previsores_treinamento, dados_classe_treinamento, dados_previsores_test, classes_test = pickle.load(f)

Regreção_Logistica = LogisticRegression(max_iter=100, random_state=1)
Regreção_Logistica.fit(dados_previsores_treinamento, dados_classe_treinamento)

previsoes = Regreção_Logistica.predict(dados_previsores_test)

pontuação = accuracy_score(classes_test, previsoes) #0.849539406345957

#remuso do algoritimo 
from yellowbrick.classifier import ConfusionMatrix, ClassificationReport
cm = ClassificationReport(Regreção_Logistica)
cm.fit(dados_previsores_treinamento, dados_classe_treinamento)
cm.score(dados_previsores_test, classes_test)
cm.poof()

#grafico da pontuação dele
cm2 = ConfusionMatrix(Regreção_Logistica)
cm2.fit(dados_previsores_treinamento, dados_classe_treinamento)
cm2.score(dados_previsores_test, classes_test)
cm2.poof()