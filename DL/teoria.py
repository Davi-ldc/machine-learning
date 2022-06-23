import math
#pesos são inicializados com aleatoriamente


#!UMA CAMADA
"""
1 peso\
2 peso-- função soma -- função de ativação (step function)
3 peso/

(1, 2 e 3 são as camdas de entrada)
pesos multiplicam os numeros de entrada
função soma só add todos os pesos
função de ativação vai verificar se o numero que ela recebeu é maior que zero,
se for ela retorna 1, se não 0

a parte da função soma e a função de ativaçãop representa 1 neuronio

Atualização dos pesos = taxa de aprendizagem * entrada * erro

"""


def step_function(x):
    if x > 0:
        return 1
    return 0


#!DUAS CAMADAS
"""
serve para problemas não linearmente separaveis 



1 peso\
2 peso-- função soma -- função de ativação (sigmoid) neuronio 1
3 peso/\
        função soma -- função de ativação (sigmoid) neuronio 2
        
Sgmoid = 1/ (1 + (e^-x))

e = constante de euler (2.718281828459045)
x = numero que a função sigmoid recebe

descida do gradiente
*serve para frz a atualização dos pesos
vai de um ponto onde o erro é alto para um ponto onde o erro é baixo
erro = resultado esperado - resultado obtido
gradiente = sgmoid(x) * (1 - sgmoid(x))


Notas sobre a função de sigmoid:
* retorna valores entre 0 e 1
se x for alto, retorna 1
se x for baixo, retorna 0
* não retorna valores negativos



atualização dos pesos = (peson * momento) + (entrada * delta * taxa de aprendizagem)
taxa de aprendizagem = 0.1 ou 0.01
forluma delta = erro * sigmoid(x) * (1 - sigmoid(x))
"""



print(math.e)

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def delta(x):
    return sigmoid(x) * (1 - sigmoid(x))

#cauculo do valor de delta
"""
função de ativação
         |
     derivada  
         |
    delta
         |
     gradiente
     
"""

#adam é uma melhoria da descida do gradiente

def rede_neural_uma_camada(x, y):
    """x e y deve ser 0 ou 1 a rede neural vai retornar 1 ou 0
    lmbrando que 0 0 = 0, 0 1 = 0, 1 0 = 0, 1 1 = 1"""
    
    peso1 = 0.5
    peso2 = 0.5
    
    soma = x * peso1 + y * peso2
    resultado = sigmoid(soma)
    if resultado > 0.7:
        return 1
    else:
        return 0
    
#BIAS
"""
*  é um valor que soma ao final que serve para calibrar o valor fina
"""

#LOSS
"""
Loss é perda, é o erro que seu modelo comete ao executar a a transformação 
se o valor esperado é 3 e a rede respondeu 1, o erro é 2

o tipo de método q ele usa pra calcular o erro pode ser configurado 
"""

#TAXA DE APRENDIZAGEM
"""
-Maior taxa de aprendizado = Maior "passo" em direção ao erro minimo local 
(com sorte o global tb) porém maior ruido, dificuldade de convergencia e 
podendo até mesmo divergir em alguns casos.

-Menor taxa de aprendizado = Menor "passo" em direção ao erro minimo local 
(mais dificilmente atinge o global) porém o aprendizado é mais estavel, 
demora mais e se existir algum ponto de convergencia o gradiente vai aproximar de 
forma mais "direta" (lembrando que mais devagar também)


altos se usam no inicio do aprendizado, baixos se usam no momento que voce quer refinar o aprendizado
ruido é como se fosse um chiado na voz gravada de uma pessoa
quanto maior o ruido, menos vc entende a voz

"""

#resumo função de ativação
"""
step function = se x > 0 então 1 , se x < 0 então 0 

signmoid retorna valores entre 0 e 1 geralmente usada para probabilidades

tanh retorna valores negativos e positivos

softmax serve para quando vc tem mais de 2 classes, retorna a probabilidade de cada classe

relu retorna valores entre 0 e infinito e é muito usada em redes neurais convolucionais
"""

def tanh(x):
    return (math.exp(x) - math.exp(-x)) / (math.exp(x) + math.exp(-x))

def softmax(x):
    return math.exp(x) / sum(math.exp(x))


def exp(x):
    return math.e ** x




#exemplo de classificação binaria usando redes neurais
#classfica se um tumor é benigno ou maligno

import keras
from sklearn.model_selection import train_test_split
import pandas as pd

with open('dataDL/entradas_breast.csv', 'r') as f:
    variaveis_previsoras = pd.read_csv(f)

with open('dataDL/saidas_breast.csv', 'r') as f:
    classes = pd.read_csv(f)

variaveis_previsoras_treinamento, variaveis_previsoras_teste, classes_treinamento, classes_teste = train_test_split(variaveis_previsoras, classes, test_size=0.3, random_state=0)
