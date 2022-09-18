import numpy as np
import torch

# é basicamente um artificio matematico vc vai pegar uma lista q tá assim: [[4 5 6]] e deixar ela
# assim: [[4]
#  [5]
#  [6]]

# pq? pq pra frz algumas opereções com matrixes precisamos q ela estaja com um shape especifico
# e com isso vc muda o shape do vector ou da matrix sem alterar o sentido ou a informação

vector = np.array([[4,5,6]])
print(vector)
print(vector.shape)
print(vector.T)
print(np.transpose(vector))

"""
[[4 5 6]]
(1, 3)
[[4]
 [5]
 [6]]
"""
print(end="""
      
      

      """)

matrix = np.array([
      [4,5,6],
      [7,8,9]])
print(matrix)
print(matrix.shape)
print(matrix.T)
print(np.transpose(np.transpose(matrix)))#repara q oT(T(x)) = x

#       [[4 5 6]
#  [7 8 9]]
      
# (2, 3)

# [[4 7]
#  [5 8]
#  [6 9]]

#       [[4 5 6]
#  [7 8 9]]

#em torch seria 

vec = torch.tensor([[1,2,3]])
print(vec.shape)
print(vec.T)