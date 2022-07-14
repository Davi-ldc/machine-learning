#https://www.cs.cmu.edu/~aharley/vis/conv/flat.html
#redes neurais convolucionais ou CNN servem  pra Visão computacional
#não usa todas as entradas (piexels) pois é MT pixel e tem pc q aguente olhar 8 k d pixels pra classificar uma imagem em 8k
#tenta achar as caracteristicas mais importantes 
#vai selecionar somenta as melhores caracteristicas de forma que não seja necessário
#durante o treinamento descobre qual é o melhor detector de caracteristicas(usado ono operador de convolução)
#OLHAR TODOS OS PIXELS



#o que é Convolução?
#conseito ilegivel:
"""
processo de add cada elemento da imegem para seus vizinhos , ponderados por um kernel
o q vc vai frz é multiplicar a imagem pelo kernel
se vc quiser tentar entender a matematica, tem um arquivo d 3k e 500 linhas do keras 
so d cauculo sobre convolução
e tem esse link tmb: https://en.wikipedia.org/wiki/Convolution
"""

#conseito humanamente compressível:

"""
pega 8k x 3 d pixels extrais as caracteristicas de cada classe e passa dados especificos pras camdas ocultas
exemplo:
vc tem uma imagem d 32 x 32 pixels e vc quer descobrir qual o numero escrito na imagem, suponda q ela tena o formato
quadrado vc n vai precisar d tds os pixels, geralmente são usados das bprdas pra dentro, pq tipo vc n precisa
do cenario pra reconhecer a logo do google só da borda pra dentro 
Durante o treinamneto ela descobre qual é o melhor jeito d reduzir a imagem
"""










#as 4 etapas de uma rede neural convolucional:



#operação de convolução
"""
vc vai pegar os dados mandar eles prum kernel q vai modificar a imagem (reduzir a dimensinalidade da imagem)
qunato maior o valor dentro da matrix gerada pelo operação de convolução mais importante ele é
"""

#pooling
"""
reduz ainda mais a imagem(reduz o overfitting e ruidos desnecessarios)
pega uma are d 2x2 do resultado do operador d convolução e deixa somente o maior valor
multiplica a imagem pelo detector d caracteristicas q é atualizado a cada interação da rede neural junto com os pesos
da rede neural densa
"""

#flattening
"""
pega o resultado do pooling e transforma em um vetor que será passado para uma rede neural densa

"""

#rede neural densa
"""
só cria rede neural, usa relu dps softmax se for mais d 2 classes se n usa softmax
"""