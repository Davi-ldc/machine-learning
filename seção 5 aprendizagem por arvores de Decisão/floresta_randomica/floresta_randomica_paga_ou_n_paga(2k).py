import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

with open('data/paga_ou_n_paga.pkl', 'rb') as f:
    dados_previsores_treinamento, dados_classe_treinamento, dados_previsores_test, classes_test = pickle.load(f)
#0 = pagou o imprestimo
#1 = não pagou

floresta_randomica = RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=0)
#                objeto                     numero de arvores   cauculo(entropia ou impureza de gini)  faz com que td vez que agnt rode o pragrama ele apresente o mesmo resultado

floresta_randomica.fit(dados_previsores_treinamento, dados_classe_treinamento) #treina

previsoes = floresta_randomica.predict(dados_previsores_treinamento)

pontuação = accuracy_score(previsoes, dados_classe_treinamento) #MDS ELE ACERTA 100 PORCENTO DOS DADOS👀👀👀
print(pontuação)


from yellowbrick.classifier import ConfusionMatrix, ClassificationReport
#resumo do algoritimo
cm = ClassificationReport(floresta_randomica)
cm.fit(dados_previsores_treinamento, dados_classe_treinamento)
cm.score(dados_previsores_test, classes_test)
cm.poof()

#quanto ele acertou de cada classe
# cm = ConfusionMatrix(floresta_randomica)
# cm.fit(dados_previsores_treinamento, dados_classe_treinamento)
# cm.score(dados_previsores_test, classes_test)
# cm.poof()
 