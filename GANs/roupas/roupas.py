import torch
import pdb
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import FashionMNIST
from torchvision.utils import make_grid
from tqdm.auto import tqdm #barra d progreço durante o treinamento


def sow(tensor, chennels, size=(28,28), qnts_imgs=16):
    #o tensor do treinamento vai ser igual batch_size x 784 (pq 28*28 da 784)
    data = tensor.detach().cpu().view(-1, chennels, *size)#ai esse linha trasforma o tensor no formato de imagem
    #.cpu é pra usar a cpu pra processar 
    #view -1 achata o tensor tipo:
    """
    x = torch.randn(2, 3, 4)
    print(x.shape)
    > torch.Size([2, 3, 4])
    x = x.view(-1)
    print(x.shape)
    > torch.Size([24]) 24 = 2*3*4
    """#ai ele vai ter o tamnho de suas dimenções multiplicadas
    #dps disso o tamnho d data vai ser batch_size x 1 x 28 x28
    
    grid = make_grid(data[:qnts_imgs], nrow=4).permute(1,2,0)
    #             data da posição 0 a posição qnts_imgs
    # amtes da função permute fica assim: 
    #batch_size-qnts_imgs x 1 x 28 x28
    #mas pra variar pro matplot lib o numero de canais tenq vir dps do 28x28 ai dps d usar .permute fica:
    #28x28x1
    #função permute:
    """
    pega esse exemplo:
    x = torch.randn(2, 3, 5)
    print(x)
    x.size()
    torch.Size([2, 3, 5])
    >>> torch.permute(x, (2, 0, 1)).size() 
    ele ta fezendo D.size = D[2], D[0], D[1]
    torch.Size([5, 2, 3])
    """
    plt.imshow(grid)
    plt.show()


epcas = 300
cur_step = 0 
save_interval = 300

mean_g_loss = 0
mean_d_loss =0

tamnho_do_ruido = 64
lr = 0.00001

loss_fn = nn.BCEWithLogitsLoss()
#BCELoss = binary cross entropy

batch_size = 128
device = 'cuda'

dataloader = DataLoader(FashionMNIST('.', download=True, transform=transforms.ToTensor()), shuffle=True, batch_size=batch_size)
# . é onde eu vou salvar os dados
#transforms.ToTensor() transforma as imgs em um tensor
#shuffle=True faz com que agnt sempre troque a ordem das imgs durante o treinamento

def g_block(inp, out):
    return nn.Sequential(
        nn.Linear(inp, out),#linear ==Dense
        nn.BatchNorm1d(out),
        nn.ReLU(inplace=True)
    )
    
    
class Generator(nn.Module):
    def __init__(self, tamnho_do_ruido=64, tamanho_output=128, tamnho_da_img=784):
        super().__init__()
        self.Gerador = nn.Sequential(
            g_block(tamnho_do_ruido,tamanho_output),#64, 128
            g_block(tamanho_output, tamanho_output*2),#128, 256
            g_block(tamanho_output*2, tamanho_output*4),#256 512
            g_block(tamanho_output*4, tamanho_output*8),#512 1024
            nn.Linear(tamanho_output*8, tamnho_da_img),#1024, 784
            nn.Sigmoid()
        )
        
    def forward(self, noise):
            return self.Gerador(noise)
        
def get_noise(batch_size, size):
    return torch.randn(batch_size, size).to(device)
    #                 os paramentros d torch.randn é o tamnho do tesor tipo se for 2, 28,28
    #o tamnho do tensor vai ser 2x28x28
    
#---------------------------------------------------------------------------------------------------------------------
def d_block(inp, out):
    return nn.Sequential(
        nn.Linear(inp, out),
        nn.LeakyReLU(0.2),        
    )

class Descriminator(nn.Module):
    def __init__(self, tamnho_da_img=784, tamanho_output=256):
        super().__init__()
        self.Descriminador = nn.Sequential(
            d_block(tamnho_da_img, tamanho_output*4),#784, 1024
            d_block(tamanho_output*4, tamanho_output*2),#1024, 512
            d_block(tamanho_output*2, tamanho_output),#512, 256
            nn.Linear(tamanho_output, 1)#256 1
            
            
        )
    def forward(self, img):
        return self.Descriminador(img)


gen = Generator(tamnho_do_ruido).to(device)
gen_opt = torch.optim.Adam(gen.parameters(), lr=lr)

des = Descriminator().to(device)
des_opt = torch.optim.Adam(des.parameters(), lr=lr)


print(gen, des)

x,y = next(iter(dataloader))
#x é o previsor e o y é a classe

#a explicação dessa linha ta aq:
"""
# list of vowels
phones = ['apple', 'samsung', 'oneplus']
phones_iter = iter(phones)
print(next(phones_iter))   
print(next(phones_iter))    
print(next(phones_iter))    

# Output:
# apple
# samsung
# oneplus
"""

