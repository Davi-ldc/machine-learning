import pandas as pd
import numpy as np

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

print(variaveis_previsoras) # .values transforma o valor em uma array
print(type(variaveis_previsoras))# numpy.ndarray
print(type(dados)) # pandas.core.frame.DataFrame

classe = dados.iloc[:, 4].values
print(classe, end='\n\n\n\n\n')
# obs eu dividi os dados em 2 variaveis pq o algoritimo vai retornar 0 ou 1 dependendo das variaveis promissoras

maior_renda = variaveis_previsoras[:, 0].max() # 0 = income(renda) 1 = age 2 loan(divida)
menor_renda = variaveis_previsoras[:, 0].min() 
menor_idade = variaveis_previsoras[:, 1].min()
maior_idade = variaveis_previsoras[:, 1].max()
menor_divida = variaveis_previsoras[:, 2].min()
maior_divida = variaveis_previsoras[:, 2].max()
print(maior_renda, menor_renda,
      
      maior_divida, menor_divida,
      
      maior_idade,
      menor_idade)