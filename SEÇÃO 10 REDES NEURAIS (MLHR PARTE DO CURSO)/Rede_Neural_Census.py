import pickle 
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score


with open('data/census.pkl', 'rb') as f:
    variaveis_previsoras_treinamento, classes_treinamento, variaveis_previsoras_teste, classes_teste = pickle.load(f)


rede_neural = MLPClassifier(max_iter=1000, tol=0.0001)
#                       maximo de iteracoes, tolerancia(se a rede nao melhorar mais do que a tolerancia o treinamento para)

rede_neural.fit(variaveis_previsoras_treinamento, classes_treinamento)