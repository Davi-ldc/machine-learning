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
    
      ## optional
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
    
    