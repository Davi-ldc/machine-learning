import torch, torchvision, os, PIL, pdb
import numpy as np
import matplotlib.pyplot as plt
from torch import nn 
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import make_grid
from tqdm.auto import tqdm
from PIL import Image

batch_size = 128
tamnho_do_ruido = 200
lr=1e-4

def show(tensor, qnts_imgs=25, wandbactive=0, name=''):
    data = tensor.detach().cpu()
    grid = make_grid(data[:qnts_imgs], nrow=5).permute(1,2,0)
    #             data da posição 0 a posição qnts_imgs
    #make grid pega varias imgs e retorna 
    # amtes da função permute fica assim: 
    #batch_size-qnts_imgs x 1 x 28 x28
    #mas pra variar pro matplot lib o numero de canais tenq vir dps do 28x28 ai dps d usar .permute fica:
    #28x28x1
    #função permute:
    """

    x = torch.randn(2, 3, 5)
    print(x)
    x.size()
    torch.Size([2, 3, 5])
    >>> torch.permute(x, (2, 0, 1)).size() 
    ele ta fezendo D.size = D[2], D[0], D[1]
    torch.Size([5, 2, 3])
    """
    


        
    #clip(0,1) significa que
    """
    If any element in the grid is
    <0
    It will be set to 0 
    If any element is > 1
    It will be set to 1
    """
    
    plt.imshow(grid.clip(0,1))
    plt.show()

    
class Gerador(nn.Module):
    def __init__(self, tamnho_do_ruido=64, output=16):
        super(Gerador, self).__init__()
        self.tamnho_do_ruido = tamnho_do_ruido

        self.gen = nn.Sequential(

            nn.ConvTranspose2d(tamnho_do_ruido, output * 32, 4, 1, 0), ## 4x4
            #                 entrada      saida    tamnho_d_kernel
            nn.BatchNorm2d(output*32),
            nn.ReLU(True),

            nn.ConvTranspose2d(output*32, output*16, kernel_size=4, stride=2, padding=1), ## 8x8
            nn.BatchNorm2d(output*16),
            nn.ReLU(True),

            nn.ConvTranspose2d(output*16, output*8, 4, 2, 1), ## 16x16
            nn.BatchNorm2d(output*8),
            nn.ReLU(True),

            nn.ConvTranspose2d(output*8, output*4, 4, 2, 1), ## 32x32     
            nn.BatchNorm2d(output*4),      
            nn.ReLU(True),            

            nn.ConvTranspose2d(output*4, output*2, 4, 2, 1), ## 64x64    
            nn.BatchNorm2d(output*2),             
            nn.ReLU(True),            

            nn.ConvTranspose2d(output*2, 3, 4, 2, 1), ## 128x128    
            #             entrada   3=saida 4= tmnh do kernel, 2 = strides 1 = paddding     
            nn.Tanh()# -1 a 1
        )
        
    def forward(self, noise):
        x = noise.view(len(noise), self.tamnho_do_ruido, 1, 1)# 128 x 200 x 1 x 1
        return self.gen(x)
gen = Gerador(tamnho_do_ruido).to('cpu')
gen_opt = torch.optim.Adam(gen.parameters(), lr=lr, betas=(0.5, 0.9))
def get_noise(n, tamnho_do_ruido, device='cpu'):
    return torch.randn(n, tamnho_do_ruido, device=device)

#### Generate new faces

checkpoint = torch.load('G-latest (6).pkl', map_location=torch.device('cpu'))
gen.load_state_dict(checkpoint['model_state_dict'])
gen_opt.load_state_dict(checkpoint['optimizer_state_dict'])

noise = get_noise(batch_size, tamnho_do_ruido)
fake = gen(noise)
show(fake)
plt.imshow(fake[16].detach().cpu().permute(1,2,0).squeeze().clip(0,1))

