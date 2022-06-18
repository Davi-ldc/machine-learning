#Pega o nlp como exemplo os dados estão super desbalanciados 
#o que faz com a pontuação dele numa classe sejá incrivel e en outra seja patetico
#pra resover isso agnt pode add mais dados na coluna menor ou tirar dados da coluna maior

#ex_tirando_dados:
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import TomekLinks

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


tl = TomekLinks(sampling_strategy='majority')
variaveis_previsoras, classes = tl.fit_resample(variaveis_previsoras, classes)

print(variaveis_previsoras.shape, classes.shape)