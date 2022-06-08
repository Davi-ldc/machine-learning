import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


with open('data/census.pkl', 'rb') as f:
    dados_previsores_treinamento, classes_treinamento, dados_previsores_test, classes_teste = pickle.load(f)


floresta_randomica = RandomForestClassifier(n_estimators=200, criterion='entropy', random_state=0)
#               objeto                     numero de arvores   cauculo(entropia ou impureza de gini)  faz com que td vez que agnt rode o pragrama ele apresente o mesmo resultado

floresta_randomica.fit(dados_previsores_treinamento, classes_treinamento)

previsoes = floresta_randomica.predict(dados_previsores_test)

pontuação = accuracy_score(previsoes, classes_teste) # 0.8501535312180143
print(pontuação)

#remuso do algoritimo 
from yellowbrick.classifier import ConfusionMatrix, ClassificationReport
cm = ClassificationReport(floresta_randomica)
cm.fit(dados_previsores_treinamento, classes_treinamento)
cm.score(dados_previsores_test, classes_teste)
cm.poof()

#grafico da pontuação dele
cm2 = ConfusionMatrix(floresta_randomica)
cm2.fit(dados_previsores_treinamento, classes_treinamento)
cm2.score(dados_previsores_test, classes_teste)
cm2.poof()

#obs com 10 arvores o percentual de acerto é de 83 porcento e com 100 arvores ele é de 85 porcento e com 200 arvores ele tembem é de 85 porcento
#ou seja n copensa aumentar muito o numero de arvores-