import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

with open('data/risco_credito.pkl', 'rb') as f:
    dados_previsores, classes = pickle.load(f)
    
dados_previsores = np.delete(dados_previsores, 2, 7, 11, axis=0)
#                                                indices  1 apaga colunas   0 apaga linhas
classes = np.delete(classes, 2, 7, 11, axis=0)

regreção_lojistica = LogisticRegression()
