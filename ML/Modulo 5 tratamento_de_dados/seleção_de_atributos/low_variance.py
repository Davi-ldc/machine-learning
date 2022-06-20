import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


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


from sklearn.feature_selection import VarianceThreshold

seleção = VarianceThreshold(threshold=0.22)
#theshold = é a variancia minima
seleção.fit(variaveis_previsoras)
variaveis_previsoras = seleção.transform(variaveis_previsoras)
print(seleção.variances_)