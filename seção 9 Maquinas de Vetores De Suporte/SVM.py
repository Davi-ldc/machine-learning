#bom algoritimo para problemas complexos como reconhecimento de voz, caracteres, imagens, etc.

#antes das resdes neurais o svm era considerado o melhor algoritimo de ML

#aprende hiperplanos de separação com margem maxima 
#tem como objetivo encontrar a melhor reta possivel para os dados

"""
Formula:

1/2 ||w||^2 + c ∑ai
                i
c = penalização (enquanto maior, melhor)
"""


#LINEAR X NÃO LINEAR
"""
SVMs não lineaveis(não é possivel trassar uma reta que separe os dados) -> Kernel Trick(faz  com que seja possivel trasar uma reta que separe os dados)
SVMs lineaveis(possivel trassar uma reta que separe os dados) -> Linear SVM

Tipos de Kernel:
Linear: K(x,y) = xy
Gaussiano: K(x,y) = exp(-||x-y||^2/2σ^2)
Polynomial: K(x,y) = (γx.y + c)^d
Tangente Hiperbolica: K(x,y) = tanh(γx.y + c)
"""