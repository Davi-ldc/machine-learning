import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

with open('data/risco_credito.pkl', 'rb') as f:
    dados_previsores, classes = pickle.load(f)
    
dados_previsores = np.delete(dados_previsores, [2, 7, 11], axis=0)
#                                                indices  1 apaga colunas   0 apaga linhas
classes = np.delete(classes, [2, 7, 11], axis=0)

regreção_lojistica = LogisticRegression(max_iter=100, random_state=1)
#                         numero maximo de ajustes no cauculo do algoritimo         faz com que os resultados sejam sempre iguais

regreção_lojistica.fit(dados_previsores, classes)

print(regreção_lojistica.intercept_)#coeficiente de b0
print(regreção_lojistica.coef_)# coeficientes

previzoes = regreção_lojistica.predict([[0,0,1,2], [2,0,0,0]])
print(previzoes)