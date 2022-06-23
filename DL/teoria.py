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
* é um
"""



#resumo função de ativação
"""
step function = se x > 0 então 1 , se x < 0 então 0 

signmoid retorna valores entre 0 e 1 geralmente usada para probabilidades
"""