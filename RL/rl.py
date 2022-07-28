#ação q tem o maior valor d Q é a escolida
"""
funciona assim: vai ter uma ia q vai estar em um ambiente ai o objetivo dela 
é aprender as ações que dão o maximo de recompensa.
é baseado no jeito q os humanos aprendem, se botar a mão no fogo TOMA recompensa negativa
Se comer coisa boa ganha recompensa positiva, funciona muito bem
e fica mlhr ainda com GANs
é basicamente tentaiva e erro, se ele ganha ponto no estado de comer
então onde ele vai atribuir 1 a comida
se ele perde ponto no estado d queimar ele bota um -1 no fogo
ai qnd ele tiver q decidir entre fogo e comida ele vai "lembrar" da vez q ele morreu queimado
e da q ele ficou feliz comendo
o problema é que se ele tiver a opção de comer e comer, ele n vai saber oq frz pq os dois tem
valores iguais ai pra resolver isso existe a equação de bellman
"""

#Equação de Bellman
"""
s = estado
a = ação
r = recompensa
y = fator de desconto

V(s) = max(R(s,a)+yV(s'))
        a

Enquanto maior a recompensa mais ariscado é ia

traduzindo:

V(s) = max(R(s,a)+yV(s'))

VALOR_ESPERADO = MAIOR VALOR( RECOMPENSA(estado; ação) + y * RECOMPENSA(estado anterior;ação))

o valor esperado é igual o maior valor de recompensa que podemos ter, somado de uma parcela da recompensa do estado anterior+

"""

#processo de markov (cm a ia vai tomar a decisão)
"""
pra cada celula vc vai ter um numero Q(s, a)
pra tomar a decisao vc escolhe o maior Q que deve ser a maior recompensa
depois de tomar a decisao, o algoritmo recebe a resposta se ele "acertou" ou nao, e faz o ajuste no Q(s, a)
dai na proxima decisao, vc escolhe novamente o maior Q
e faz o ajuste de novo
de forma resumida e basica é isso
uma tabelinha que diz
Se voce estiver no estado 'a1' (de frente pra uma parede por exemplo) -> vire pra direita (Q=1) ou vire pra esquerda(Q=0.3). 1 é maior que 0.3 entao vire pra direita
se voce estiver no estado 'a2' (de costas pra parede) -> siga em frente (Q=0.8) ou vire pra direita (Q=0.5) ou vire pra esquerda (Q=0.7). 0.8 é o maior entao siga em frente
"""


#loss
"""
l =somatorio(Q-target - q)^2
(diferença do ganho esperado pro ganho real)
"""
