import pickle
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree


with open('data/risco_credito.pkl', 'rb') as f:
    dados_previsores, classes = pickle.load(f)

arvore = DecisionTreeClassifier(criterion="entropy", random_state=0)
#                                  cauculo vai usar entropia

arvore.fit(dados_previsores, classes)#treino

print(dados_previsores)
print(arvore.feature_importances_) # mostra os dados previsores e seu nivel de importancia

nome_dos_previsores = ['história', 'dívida', 'garantias', 'renda']
#nome dos dados previsores
tree.plot_tree(arvore, feature_names=nome_dos_previsores, class_names=arvore.classes_,  filled=True);
#                nomes dos previsores     nome das classes               botar cor nas caixas     faz com que não aparessa texto (; ele descreve a arvore em forma d texto)
plt.show() #mostra a arvore

# previsoes = arvore.predict([[0,0,1,2]])
# print(previsoes)

