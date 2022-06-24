import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# with open('data/census.csv') as f:
#     data = pd.read_csv(f)
    
# variaveis_previsoras = data.iloc[:, 0:14].values
# classes = data.iloc[:, 14].values

# label = LabelEncoder() 
# for c in range(1, 14): 
#     if c == 2 or c == 10 or c == 11 or c == 12: # colunas que ja sao numericas 
#         continue 
#     variaveis_previsoras[:, c] = label.fit_transform(variaveis_previsoras[:, c]) 
    
#     Q_numero_cada_str_recebeu = dict(zip(label.classes_, label.transform(label.classes_)))
#     # print(Q_numero_cada_str_recebeu)

# variaveis_previsoras_treino, variaveis_previsoras_teste, classes_treino, classes_teste = train_test_split(variaveis_previsoras, classes, test_size=0.3, random_state=0)

# #converte pra df
# pd.DataFrame(variaveis_previsoras_treino).to_csv('dataDL/variaveis_previsoras_treino.csv', index=False)
# pd.DataFrame(variaveis_previsoras_teste).to_csv('dataDL/variaveis_previsoras_teste.csv', index=False)
# pd.DataFrame(classes_treino).to_csv('dataDL/classes_treino.csv', index=False)
# pd.DataFrame(classes_teste).to_csv('dataDL/classes_teste.csv', index=False)
 
 
#carega os dados
variaveis_previsoras_treino = pd.read_csv('dataDL/variaveis_previsoras_treino.csv')
variaveis_previsoras_teste = pd.read_csv('dataDL/variaveis_previsoras_teste.csv')
classes_treino = pd.read_csv('dataDL/classes_treino.csv')
classes_teste = pd.read_csv('dataDL/classes_teste.csv')
 

 