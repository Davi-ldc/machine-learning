import pickle
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

with open('data/census.pkl', 'rb') as f:
    dados_previsores_treinamento, classes_treinamento, dados_previsores_test, classes_teste = pickle.load(f)
    
arvore = DecisionTreeClassifier(criterion="entropy", random_state=0)

#treina
arvore.fit(dados_previsores_treinamento, classes_treinamento)

previsoes = arvore.predict(dados_previsores_test)

pontuação = accuracy_score(previsoes, classes_teste) #0.8104401228249745


#mostra a arvore
# tree.plot_tree(arvore,  filled=True);
#                   nome dos previsores         nome das classes (tenq ser uma str)   bgcolor=true 
# plt.show() é tão grande q fica impossivel d ver

#grafico da pontuação dele
from yellowbrick.classifier import ConfusionMatrix
cm = ConfusionMatrix(arvore)
cm.fit(dados_previsores_treinamento, classes_treinamento)
cm.score(dados_previsores_test, classes_teste)
cm.poof()

#resumo da arvore
from yellowbrick.classifier import ClassificationReport
cm = ClassificationReport(arvore)
cm.fit(dados_previsores_treinamento, classes_treinamento)
cm.score(dados_previsores_test, classes_teste)
cm.poof()