import pandas as pd
import numpy as np


dados = pd.read_csv('lerCVS/dados.csv')
"""imagina que vc fez uma pesquisa sobre a idade da pessoas em um determinado local
alguem pode ter respondido zuando dizendo q tem idade negativa ou q nasceu antes do bigbang
ai vc tenq filtrar esses dados pra iginorar quem repondeu a pesquisa zuando
EX:"""

#filtros:
print(dados.loc[dados['age'] < 0])
print(dados.loc[(dados['clientid'] == 16) | (dados['clientid'] == 22) | (dados['clientid'] == 27)])
# | == or mas no pandas se vc or da erro entçao usa o |

