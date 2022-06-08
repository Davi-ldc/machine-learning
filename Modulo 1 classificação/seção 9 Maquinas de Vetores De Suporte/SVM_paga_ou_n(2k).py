import pickle 
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

with open('data/paga_ou_n_paga.pkl', 'rb') as f:
    dados_previsores_treinamento, dados_classe_treinamento, dados_previsores_test, classes_test = pickle.load(f)
#0 = pagou o imprestimo
#1 = não pagou

svm = SVC(kernel ='rbf', C = 2.0, random_state = 1)
#        kernel        penalização(quanto maior mlhr)    random_state=1 faz com que o algoritimo seja sempre o mesmo

svm = svm.fit(dados_previsores_treinamento, dados_classe_treinamento)

previzões = svm.predict(dados_previsores_test)

pontuação = accuracy_score(classes_test, previzões) #0.988
print(pontuação)

#remuso do algoritimo 
from yellowbrick.classifier import ConfusionMatrix, ClassificationReport
cm = ClassificationReport(svm)
cm.fit(dados_previsores_treinamento, dados_classe_treinamento)
cm.score(dados_previsores_test, classes_test)
cm.poof()

#grafico da pontuação dele
cm2 = ConfusionMatrix(svm)
cm2.fit(dados_previsores_treinamento, dados_classe_treinamento)
cm2.score(dados_previsores_test, classes_test)
cm2.poof()