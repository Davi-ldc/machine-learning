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
        
s' é o estado anterior 
s é o estado atual

"""