import pickle
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score


with open('data/census.pkl', 'rb') as f:
    variaveis_previsoras_treinamento, classes_treinamento, variaveis_previsoras_teste, classes_teste = pickle.load(f)


rede_neural = MLPClassifier(max_iter=1000, verbose=True, hidden_layer_sizes = (120, 120, 120), tol=0.000000001)
#max_iter = maximo de iteracoes, 
# tol = tolerancia(se a rede nao melhorar mais do que a tolerancia o treinamento para)
#verbose=True, mostra o progresso do treinamento

rede_neural.fit(variaveis_previsoras_treinamento, classes_treinamento)

previzões = rede_neural.predict(variaveis_previsoras_teste)

pontuação = accuracy_score(classes_teste, previzões) #0.8178096212896623
print(pontuação)

from yellowbrick.classifier import ConfusionMatrix, ClassificationReport

cm = ClassificationReport(rede_neural, classes=['<=50K', '>50K'])
cm.fit(variaveis_previsoras_treinamento, classes_treinamento)
cm.score(variaveis_previsoras_teste, classes_teste)
cm.poof()