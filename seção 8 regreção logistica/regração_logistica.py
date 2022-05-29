#antes d falar sobre o algoritimo é importante lembrar:

"""
    METODOS PREDITIVOS    |   METODOS DESCRITIVOS
    classificação(rotulos)|   Associação
    regreção(numeors)     |   Agrupamento
                          |   Detecção de desvios
                          |   Padrões sequenciais
                          |   Sumarização

ex de regreção:
qnd vc quer prever o limite de um cartão d credito ele n pode ser um rotulo (alto medio baixo)
tenq ser um numero, pra isso serve regreção
"""

#objetivo do algoritimo:
#encontrar a melhor linha de regreção para aquele conjunto de dados


#formula (função sigmoid)

"""
p = 1 / (1 + e^(-y))
e = numero de Euler
y = b0 + b1*x
"""
import math

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

#no trinamento, o algoritimo vai tentar prever o mlhr valor de y

