import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

with open('data/paga_ou_n_paga.pkl', 'rb') as f:
    dados_previsores_treinamento, dados_classe_treinamento, dados_previsores_test, classes_test = pickle.load(f)
#0 = pagou o imprestimo
#1 = não pagou
    
    
knn = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
#                      numero d visinho      metrica    tipo de cauculo (distancia euclidiana ou Geometria do táxi)
knn.fit(dados_previsores_treinamento, dados_classe_treinamento) #treina

previsoes = knn.predict(dados_previsores_test)


from sklearn.metrics import accuracy_score

pontuação = accuracy_score(classes_test, previsoes) # 0.986
print(pontuação)

#remuso do algoritimo 
from yellowbrick.classifier import ConfusionMatrix, ClassificationReport
cm = ClassificationReport(knn)
cm.fit(dados_previsores_treinamento, dados_classe_treinamento)
cm.score(dados_previsores_test, classes_test)
cm.poof()

#grafico da pontuação dele
cm2 = ConfusionMatrix(knn)
cm2.fit(dados_previsores_treinamento, dados_classe_treinamento)
cm2.score(dados_previsores_test, classes_test)
cm2.poof()
