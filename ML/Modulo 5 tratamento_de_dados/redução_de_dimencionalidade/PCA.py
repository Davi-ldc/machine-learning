#Principal component analysis (PCA) (não supervisionado)
#é um método de dimensionamento de dados que utiliza
#a projeção de componentes principais para reduzir a dimensão de um conjunto de dados.

#identifica a corelação entre as variaveis que, caso seja forte, é possivel reduzir
#a dimencionalidade

#quando o problema é mais complexo ou os dados não saõ linearmente separaveis
#vc pd usar o kernel PCA


#exemplo:


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

    
pca = PCA(n_components=2)

menos_atributos_treinamento = pca.fit_transform(variaveis_previsoras_treino)
menos_atributos_teste = pca.transform(variaveis_previsoras_teste)
#os atributos mais corelacionados serão unidos

floresta = RandomForestClassifier(n_estimators=100, random_state=0)
floresta.fit(menos_atributos_treinamento, classes_treino)
previsoes = floresta.predict(menos_atributos_teste)

pontuação = accuracy_score(classes_teste, previsoes)
print(pontuação)


# kernel_pca = KernelPCA(n_components=2, kernel='rbf')
# menos_atributos_treinamento = kernel_pca.fit_transform(variaveis_previsoras_treino)
# menos_atributos_teste = kernel_pca.transform(variaveis_previsoras_teste)
    
# floresta = RandomForestClassifier(n_estimators=100, random_state=0) 
# floresta.fit(menos_atributos_treinamento, classes_treino)
# previsoes = floresta.predict(menos_atributos_teste)
    
# pontuação = accuracy_score(classes_teste, previsoes)
# print(pontuação)