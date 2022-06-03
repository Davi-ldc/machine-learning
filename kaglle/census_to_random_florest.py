import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

with open('data/census.csv') as f:
    data = pd.read_csv(f)
    
variaveis_previsoras = data.iloc[:, 0:14].values
classes = data.iloc[:, 14].values

label = LabelEncoder() 
for c in range(1, 14): 
    if c == 2 or c == 10 or c == 11 or c == 12: # colunas que ja sao numericas 
        continue 
    variaveis_previsoras[:, c] = label.fit_transform(variaveis_previsoras[:, c]) 
    
    Q_numero_cada_str_recebeu = dict(zip(label.classes_, label.transform(label.classes_)))
    print(Q_numero_cada_str_recebeu)

variaveis_previsoras_treino, variaveis_previsoras_teste, classes_treino, classes_teste = train_test_split(variaveis_previsoras, classes, test_size=0.3, random_state=0)

floresta = RandomForestClassifier(n_estimators=100, criterion='entropy', random_state=0)

floresta.fit(variaveis_previsoras_treino, classes_treino)

previzões = floresta.predict(variaveis_previsoras_teste)
pontuação = accuracy_score(previzões, classes_teste)#0.8575084450813799
print(pontuação)


from yellowbrick.classifier import ConfusionMatrix, ClassificationReport

cm = ClassificationReport(floresta)
cm.fit(variaveis_previsoras_treino, classes_treino)
cm.score(variaveis_previsoras_teste, classes_teste)
cm.poof()

#quanto ele acertou de cada classe
cm2 = ConfusionMatrix(floresta)
cm2.fit(variaveis_previsoras_treino, classes_treino)
cm2.score(variaveis_previsoras_teste, classes_teste) 
cm2.poof()
 