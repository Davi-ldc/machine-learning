import pickle 
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


with open('data/census.pkl', 'rb') as f:
    variaveis_previsoras_treinamento, classes_treinamento, variaveis_previsoras_teste, classes_teste = pickle.load(f)
    
svm = SVC(C=2.0, random_state=1)

svm.fit(variaveis_previsoras_treinamento, classes_treinamento)

previzões = svm.predict(variaveis_previsoras_teste)


pontuação = accuracy_score(classes_teste, previzões)
print(pontuação)


# #remuso do algoritimo 
# #remuso do algoritimo 
# from yellowbrick.classifier import ConfusionMatrix, ClassificationReport
# cm = ClassificationReport(svm)
# cm.fit(variaveis_previsoras_treinamento, classes_treinamento)
# cm.score(variaveis_previsoras_teste, classes_teste)
# cm.poof()

# #grafico da pontuação dele
# cm2 = ConfusionMatrix(svm)
# cm2.fit(variaveis_previsoras_treinamento, classes_treinamento)
# cm2.score(variaveis_previsoras_teste, classes_teste)
# cm2.poof()