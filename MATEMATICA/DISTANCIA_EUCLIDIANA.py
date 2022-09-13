"""
então..........................
Era uma vez um motorista d trem q precisava da matematica pra manter seu nogocio (ganhar carvão no natal)
Esse é "Euclides"......😐🙄

Uma vez eucides queria medir a distantacia entre 2 vetores (por mais incrivel q pareça n é inutil)
ai ele criou isso:

                          n
DE(x, y) = raiz_quadrada(somatorio(xi-yi)^2)
                          i=1

funciona assim: agnt sabe q se tem um triangulo com um angulo d 90 graus então
lado_pequeno**2 lado_medio**2 = lado_grande**
repara q nesse caso:


5|
4|   *
3|
2|
1|*
x|------------------
 y 1  2   3   4   5 

a linha entre os pontos é a distancia entre eles. Se vc puchar 2 linhas retas dos pontos elas 
se encontram e formam um triangulo



5|
4|   *
3|  /|
2| / |
1|*---
x|------------------
 y 1  2   3   4   5 

sabendo q a distancia sempre vai ser o maior lado isso vira uma equção simples

se x = [3,4]
e y = [0,0]
então fica:

(3-0)**2\ 9 
+        + 25 -> sqrt() - 5
(4-0)**2/ 16

ai o (3-0) é o lado menor e o 4-0 é o lado medio. Se n tivesse a DE ia ser:

(4-0)**2 + (3-0)**2 = D**2
16 + 9 = D**2
D**2 = 25
D = sqrt(25)
"""

#emplementação em pyhton
import math



def DE(x, y) -> float:
    result = 0
    if len(x) != len(y) or len(x) == 0 or len(y) == 0:
        raise Exception("x and y must have the same shape")
    for i in range(len(x)):
        z = (x[i] - y[i])**2
        result += z
    return math.sqrt(result)



print(DE([3, 4],[0,0]))