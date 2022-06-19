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


from sklearn.ensemble import ExtraTreesClassifier

seleção = ExtraTreesClassifier(random_state=0)
seleção.fit(variaveis_previsoras, classes)
print(seleção.feature_importances_)#lembra q arvore de decisao 
#descobre a importancia de cada atributo? 
#isso mostra qual atributo ela acha q é mais importante
#obs os valores estão em porcentagem (porcentagem de importancia para a classe)