import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
"""
ta imagina q vc vai frz uma rede neural como uma base de dados bancaria
se ouver uma diferença muiuto grande em alguns valores como a divida e a renda a rede
neural vai achar q eles são mais importantes por terem peso maior ai pra resolver isso da pra usar uma
matematica imcompreensivel OUUUUU deixar o skllear frz ela por vc
ai os valores ficam na mesma escala e todos terão o msm nivel de importancia pro algoritimo
"""


dados = pd.read_csv('pre-processamento_de_dados/data.csv') 
# age idade, 
#defaut 0 é qnd o cliente pagou o emprestimo e 1 é qnd ele n pagou
#income é renda
#loan é divida

#meu objetivo é ultilizar esse base de dados para frz uma previsão (se alguem vai ou não pagar um emprestimo)

#acha o index de valores falsos ou nulos
indice_idades_falsas = dados[dados['age'] < 0].index
indice_idades_faltando = dados.loc[pd.isnull(dados['age'])].index

#remove eles da base d dados
dados = dados.drop(indice_idades_falsas)
dados = dados.drop(indice_idades_faltando)

variaveis_previsoras = dados.iloc[:, 1:4].values
#      seleciona linas e colunas fala que queremos todas as linhas seleciona as 3 ultimas colunas (ele não vai ate o 4)
#pois não queremos q o id do usariario seja levado em consideração pelo algoritimo.
classe = dados.iloc[:, 4].values
# obs eu dividi os dados em 2 variaveis pq o algoritimo vai retornar 0 ou 1 dependendo das variaveis promissoras

padrao = StandardScaler()
variaveis_previsoras = padrao.fit_transform(variaveis_previsoras)
menor_renda = variaveis_previsoras[:, 0].min()
menor_idade = variaveis_previsoras[:, 1].min()
menor_divida = variaveis_previsoras[:, 2].min()
print(menor_renda, menor_idade, menor_divida)