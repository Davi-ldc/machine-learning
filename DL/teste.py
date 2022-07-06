import pandas as pd


previsores = pd.read_csv('dataDL/entradas_breast.csv')
classe = pd.read_csv('dataDL/saidas_breast.csv')

from sklearn.ensemble import RandomForestClassifier

floresta = RandomForestClassifier(n_estimators = 100)
floresta.fit(previsores, classe)

previsoes = floresta.predict(previsores)
from sklearn.metrics import confusion_matrix, accuracy_score

pontuação = accuracy_score(classe, previsoes)
print(pontuação)