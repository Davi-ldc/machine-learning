#a ideia é dividir sua base de dados em varios pedaços
#usar um deles para teste e 9 para treino
#ai ele faz isso 10 vezes sendo que a parte de teste é sempre diferente
#apos a 10 interação do algoritimo ele caucula a media de acerto de todas as 10 interações

import pickle
from sklearn.model_selection import cross_val_score, KFold
from sklearn.ensemble import RandomForestClassifier

with open('data/census.pkl', 'rb') as f:
    variaveis_previsoras_treinamento, classes_treinamento, variaveis_previsoras_teste, classes_teste = pickle.load(f)


k_fload = KFold(n_splits=10, shuffle=True) #divide a base de dados
# n_splits = numero de pedaços em que a base vai se dividir 
#shuffle=True é para que o algoritimo sempre misture os daodos


floresta = RandomForestClassifier(n_estimators=50, criterion='entropy')
pontuação = cross_val_score(floresta, variaveis_previsoras_treinamento, classes_treinamento, cv=k_fload)#faz os testes com a base de dados dividida
print(f'media da pontuação da arvore de decisão: {pontuação.mean()}')

# from sklearn.neural_network import MLPClassifier
# r_d = MLPClassifier(max_iter=1000, verbose=True, hidden_layer_sizes = (120, 120, 120), tol=0.000000001)
# pontuação = cross_val_score(r_d, variaveis_previsoras_treinamento, classes_treinamento, cv=k_fload)
# print(f'media da pontuação da rede neural: {pontuação.mean()}')
