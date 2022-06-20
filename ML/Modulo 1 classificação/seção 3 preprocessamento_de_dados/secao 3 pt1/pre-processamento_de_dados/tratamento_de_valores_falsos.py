import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

dados = pd.read_csv('pre-processamento_de_dados/data.csv') # defaut 0 é qnd o cliente pagou o emprestimo e 1 é qnd ele n pagou


mentirozos = dados.loc[dados['age'] < 0] # localiza as pessoas que tem idade negativa
#OBS: loc não é uma função e sim uma propriedade por isso é executado com []
print(mentirozos)

onde_ta_os_mentirozos = dados[dados['age'] < 0].index # retorna o indece das pessoas que tem idade negativa
print(onde_ta_os_mentirozos)

dados_filtrados = dados.drop(onde_ta_os_mentirozos) # cria um novo dataframe sem idades negativas

print(dados_filtrados.loc[dados_filtrados['age'] < 0], end='\n\n\n\n\n')
#mostra que ninguem tem idade negativa nessa base de dados
#----------------------------------------------------------------------------------------------------------------------




"""Agora imagagina q vc n ta com muitos dodos e vc trocar as idades falsas ao inves de deletar
todos os dados da pessoas que tem idade negativa
Pra frz isso vc pd usar a media das idades normais e botar elas com a idade de quem respondeu zuando
(geralmente fazem isso pra reaproveira os outros dados)
mas na maioria dos casos vc vai deletar todos os dados d quem respondeu zuando"""

#para achar a media de algo dentro da base de dados pode se usar o mean EX:
print(dados.mean())
#vc tmb pode aplicar filtros como o loc, drop, 4etc
media_idade = dados['age'][dados['age'] > 0].mean() #calcula a media das idades maiores que 0 normais
#       seleciona a coluna   aplica o filtro   calcula a media
dados.loc[dados['age'] < 0, 'age'] = media_idade #substitui as idades negativas pela media
    #seleciona a coluna   aplica o filtro na coluna age substitui as idades negativas pela media
    #(sem o ultimo 'age' todos os dados dessa pessoas seriam alterados pois eu n taria especificando qual dado alterar ai ele alteraria todos¯\_(ツ)_/¯)
print(dados.loc[dados['age'] < 0]) #mostra que ninguem tem idade negativa nessa base de dados já q os valores falsos foram substituidos
print(f"""a media das idades é {media_idade}""")

#----------------------------------------------------------------------------------------------------------------------

print(dados.loc[dados['age'] < 0, 'age'])



#pra visualizar uma base de dados pode se usar o parametro head EX:
print(dados.head(27))