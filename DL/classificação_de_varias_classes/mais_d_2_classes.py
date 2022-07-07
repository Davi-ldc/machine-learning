import pandas as pd

data = pd.read_csv('dataDL/iris.csv')

variaveis_previsoras = data.iloc[:, 0:4].values
variaveis_classe = data.iloc[:, 4].values