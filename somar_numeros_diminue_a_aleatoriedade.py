import matplotlib.pyplot as plt
import random

l = []
l2= []
for c in range(3000000):#quanto maior o numero do for maior é a quantidade de 1500
    n1 = random.randint(0, 1000)
    n2 = random.randint(0, 1000)
    n3 = random.randint(0, 1000)
    n = n1 + n2 + n3
    l2.append(n1)
    l.append(n)
print(len(l))

zero = 0
tres_mil = 0
mais_provavel = 0
for c in l:
    if c == 0:
        zero += 1
    if c == 3000:
        tres_mil += 1
    if c == 1500:
        mais_provavel += 1
        
print(zero, tres_mil)
print(zero+tres_mil)
print(mais_provavel)
    
#grafico com os resultados
plt.hist(l, bins=10, range=(0, 3000))
plt.show()

#grafico com l2
plt.hist(l2, bins=10, range=(0, 1000))
plt.show()