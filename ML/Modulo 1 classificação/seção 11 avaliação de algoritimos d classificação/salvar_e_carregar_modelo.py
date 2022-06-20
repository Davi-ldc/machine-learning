import pickle
import numpy as np
from sklearn.neural_network import MLPClassifier

with open('data/census.pkl', 'rb') as f:
    variaveis_previsoras_treinamento, classes_treinamento, variaveis_previsoras_teste, classes_teste = pickle.load(f)

variaveis_previsoras = np.concatenate((variaveis_previsoras_treinamento, variaveis_previsoras_teste), axis=0)
classes = np.concatenate((classes_treinamento, classes_teste), axis=0)

rede_neural = MLPClassifier(max_iter=1000, verbose=True, hidden_layer_sizes = (150, 150, 150), tol=0.000000001)
#max_iter = maximo de iteracoes, 
# tol = tolerancia(se a rede nao melhorar mais do que a tolerancia o treinamento para)
#verbose=True, mostra o progresso do treinamento

rede_neural.fit(variaveis_previsoras, classes) #treina

#salva o modelo
pickle.dump(rede_neural, open('modelos/rede_neural_census.sav', 'wb'))

#carrega o modelo
# modlo_caregado = pickle.load(open('modelos/rede_neural_census.sav', 'rb'))