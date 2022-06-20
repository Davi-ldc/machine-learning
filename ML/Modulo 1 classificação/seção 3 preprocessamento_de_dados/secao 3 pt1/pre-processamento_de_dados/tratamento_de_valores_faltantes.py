import pandas as  pd
import numpy as np

"""imagina quem vc pergunta sim ou não pra alguem e ele reponde
vai terq achar algum jeito de achar qm não respondeu e deletar os dados da pessoa
pra frz isso pode se usar a função isnull
EX: """
dados = pd.read_csv('pre-processamento_de_dados/dados.csv')
#print(dados.isnull())
"""autput:
      clientid  income    age   loan  default
0        False   False  False  False    False
1        False   False  False  False    False
2        False   False  False  False    False
3        False   False  False  False    False
4        False   False  False  False    False
...        ...     ...    ...    ...      ...
1995     False   False  False  False    False
1996     False   False  False  False    False
1997     False   False  False  False    False
1998     False   False  False  False    False
1999     False   False  False  False    False
"""

# imagina q vc tem 2K d rejistros e vc quer saber qm não respondeu
#se vc usar esse metodo ele n vai mostrar tds os dados ai pra ficar mais facil d ver qm não respondeu
#da pra usar o metodo sum EX:
#print(dados.isnull().sum())
"""output:
[2000 rows x 5 columns]
clientid    0
income      0
age         3 assim sabemos q 3 pessoas não falaram a sua idade
loan        0
default     0
dtype: int64
"""


print(dados.loc[pd.isnull(dados['age'])]) # acha os indices q não tem idade