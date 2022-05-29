import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


with open('data/paga_ou_n_paga.pkl', 'rb') as f:
    dados_previsores_treinamento, dados_classe_treinamento, dados_previsores_test, classes_test = pickle.load(f)
#0 = pagou o imprestimo
#1 = não pagou
    
Regreção_logistica = LogisticRegression(max_iter=100, random_state=1)
#                         numero maximo de ajustes no cauculo do algoritimo         faz com que os resultados sejam sempre iguais

Regreção_logistica.fit(dados_previsores_treinamento, dados_classe_treinamento)# treina

print(Regreção_logistica.intercept_)#coeficiente de b0
print(Regreção_logistica.coef_)# coeficientes

previsoes = Regreção_logistica.predict(dados_previsores_test)# previção

pontuação = accuracy_score(classes_test, previsoes) # 0.946

#remuso do algoritimo 
from yellowbrick.classifier import ConfusionMatrix, ClassificationReport
cm = ClassificationReport(Regreção_logistica)
cm.fit(dados_previsores_treinamento, dados_classe_treinamento)
cm.score(dados_previsores_test, classes_test)
cm.poof()

#grafico da pontuação dele
cm2 = ConfusionMatrix(Regreção_logistica)
cm2.fit(dados_previsores_treinamento, dados_classe_treinamento)
cm2.score(dados_previsores_test, classes_test)
cm2.poof()
