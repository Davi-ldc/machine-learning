import pickle
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
#abre a base de dados
with open('data/paga_ou_n_paga.pkl', 'rb') as f:
    dados_previsores_treinamento, dados_classe_treinamento, dados_previsores_test, classes_test = pickle.load(f)
#0 = pagou o imprestimo
#1 = não pagou


arvore = DecisionTreeClassifier(criterion='entropy', random_state=0)

#treina a arvore de decisao
arvore.fit(dados_previsores_treinamento, dados_classe_treinamento)

#teste
previsões = arvore.predict(dados_previsores_test)

#pontuação 

pontução = accuracy_score(previsões, classes_test) # 98 porcento de acerto

print(classification_report(classes_test, previsões))

#visualiza a arvore 
previsores = ['icome', 'age', 'loan']
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))

tree.plot_tree(arvore, feature_names=previsores, class_names=str(arvore.classes_),  filled=True);
#                   nome dos previsores         nome das classes (tenq ser uma str)   bgcolor=true 
plt.show()

fig.savefig("arvore.png")
#salva a arvore em um arquivo

from yellowbrick.classifier import ConfusionMatrix

cm = ConfusionMatrix(arvore)
cm.fit(dados_previsores_treinamento, dados_classe_treinamento)
cm.score(dados_previsores_test, classes_test)
cm.poof()