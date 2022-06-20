#Linear Discriminant Analysis(LDA) (supervisionado)
#util qnd vc tem um grande numero de classes
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA, KernelPCA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

with open('data/census.csv') as f:
    data = pd.read_csv(f)
    
variaveis_previsoras = data.iloc[:, 0:14].values
classes = data.iloc[:, 14].values

label = LabelEncoder() 
for c in range(1, 14): 
    if c == 2 or c == 10 or c == 11 or c == 12: # colunas que ja sao numericas 
        continue 
    variaveis_previsoras[:, c] = label.fit_transform(variaveis_previsoras[:, c]) 
    
    Q_numero_cada_str_recebeu = dict(zip(label.classes_, label.transform(label.classes_)))
    # print(Q_numero_cada_str_recebeu) 

variaveis_previsoras_treino, variaveis_previsoras_teste, classes_treino, classes_teste = train_test_split(variaveis_previsoras, classes, test_size=0.3, random_state=0)


lda = LinearDiscriminantAnalysis(n_components=1)
variaveis_previsoras_treino_lda = lda.fit_transform(variaveis_previsoras_treino, classes_treino)
variaveis_previsoras_teste_lda = lda.transform(variaveis_previsoras_teste)

#n_components cannot be larger than min(n_features, n_classes - 1)


