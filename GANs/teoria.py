"""
tem um gerador que vai receber numeros aleatorios e trasformar eles em imagem, enquanto isso o
descriminador tenta saber q img é real e qual é falsa, se ele acertar o loss diminue se errar aumenta
o loss do gerador é log(1-D(G(R))) então se ele conufundir o D o loss cai se n aumenta 


"""
