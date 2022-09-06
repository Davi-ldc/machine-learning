from turtle import forward
import torch, torchvision, os, PIL, pdb, wandb
import numpy as np
import matplotlib.pyplot as plt
from torch import nn 
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import make_grid
from tqdm.auto import tqdm
from PIL import Image



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
    
    #https://wandb.ai
    if (wandbactive==1):
        wandb.log({name:wandb.Image(grid.numpy().clip(0,1))})
        
    #clip(0,1) significa que
    """
    If any element in the grid is
    <0
    It will be set to 0 
    If any element is > 1
    It will be set to 1
    """
    
    plt.imshow(grid.clip(0,1))



epocas = 10000
batch_size = 128
lr = 1e-4
tamnho_do_ruido = 200
device = 'cuda'

step_atual = 0
critic_cycles = 5 #o critc é D e ao invez d treinar uma vez o D a outra o G agnt vai treinar 5 o C/D e 1 o G
g_losses = []
critic_losses = []
save_interval = 35
save_steps = 35

wandbactive = 1

#(wandb = pesos e bias)
wandb.login(key='')

experiment_name = wandb.util.generate_id()

myrun=wandb.init(
    project="wgan",
    group=experiment_name,
    config={
        "optimizer":"adam",
        "model":"wgan gp",
        "epoch":"1000",
        "batch_size":128
    }
)


 
    
class Gerador(nn.Module):
    def __init__(self, tamnho_do_ruido=64, output=64):
        super(Gerador, self).__init__()
        self.tamnho_do_ruido = tamnho_do_ruido

        self.gen = nn.Sequential(

            nn.ConvTranspose2d(tamnho_do_ruido, output * 32, 4), ## 4x4
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
        x = noise.view()
        return self.gen(x)
    
def get_noise(n, tamnho_do_ruido, device='cuda'):
    return torch.randn(n, tamnho_do_ruido, device=device)



#Critc

class Critic(nn.Module):
    def __init__(self, output=16):
        super(Critic, self).__init__()
        
        self.critic = nn.Sequential(
            nn.Conv2d(3, output, 4, 2, 1),#64x64
            nn.InstanceNorm2d(output),
            nn.LeakyReLU(0.2),
            
            
            nn.Conv2d(output, output*2, 4, 2, 1),# 32 x 32
            nn.InstanceNorm2d(output*2),
            nn.LeakyReLU(0.2),
            
            
            nn.Conv2d(output*2, output*4, 4, 2, 1),# 16 x 16
            nn.InstanceNorm2d(output*4),
            nn.LeakyReLU(0.2),
            
            
            
            nn.Conv2d(output*4, output*8, 4, 2, 1),# 8 x 8
            nn.InstanceNorm2d(output*8),
            nn.LeakyReLU(0.2),
            
            
            nn.Conv2d(output*8, output*16, 4, 2, 1),# 4 x 4
            nn.InstanceNorm2d(output*16),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(output*16, 1, 4, 1, 0),# 4 x 4

        )

    def forward(self, img):
        crtitic_pred = self.critic(img)
        return crtitic_pred.view(len(crtitic_pred) - 1) #tensor final vai ter shape 128 1 (pra cd parte do batch uma resposta)
    


path = "celeba_gan/data.zip"
import gdown, zipfile


# load dataset
import gdown, zipfile


url = "https://drive.google.com/uc?id=1O7m1010EJjLE5QxLZiM9Fpjs7Oj6e684"
gdown.download(url, path, quiet=True)

with zipfile.ZipFile("celeba_gan/data.zip", "r") as zipobj:
    zipobj.extractall("celeba_gan")

class DataSet():
    def __init__(self) -> None:
        pass