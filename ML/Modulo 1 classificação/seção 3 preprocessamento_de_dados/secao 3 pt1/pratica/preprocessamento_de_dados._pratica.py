import pandas as pd
import numpy as np

dados = pd.read_csv('pratica/data.csv')

#print(dados.isnull().sum()) # 3 pessoas n responderam sua idade

#print(dados.loc[dados['age'].isnull()].index) # 28 30 31 com idade faltando 
# media = dados['age'][dados['age'] > 0].mean() # 40.92770044906149

# dados.loc[dados['age'] < 0, 'age'] = media

indice_idades_falsas = dados[dados['age'] < 0].index #acha o indice da idades negativas
print(f"indice das idades negativas: {indice_idades_falsas}")

indice_idades_faltando = dados.loc[pd.isnull(dados['age'])].index
print(f"indice das idades faltando:{indice_idades_faltando}", end="\n\n\n\n\n")

dados = dados.drop(indice_idades_falsas)
dados = dados.drop(indice_idades_faltando)

print(dados.head(27))
