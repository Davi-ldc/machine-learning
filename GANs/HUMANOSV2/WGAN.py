import torch, torchvision, os, PIL, pdb
import numpy as np
import matplotlib.pyplot as plt
from torch import nn 
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import make_grid
from tqdm.auto import tqdm
from PIL import Image



def show(tensor, qnts_imgs=25):
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
        x = noise.view(len(noise), self.tamnho_do_ruido, 1, 1)# 128 x 200 x 1 x 1
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
    


import gdown, zipfile
!mkdir celeba_gan

url = "https://drive.google.com/uc?id=1O7m1010EJjLE5QxLZiM9Fpjs7Oj6e684"
output = "celeba_gan/data.zip"
gdown.download(url, output, quiet=True)

with zipfile.ZipFile("celeba_gan/data.zip", "r") as zipobj:
    zipobj.extractall("celeba_gan")

class DataSet(Dataset):
    def __init__(self, path, size=128, n_imgs = 10000):
        self.sizes = [size, size]
        itens, labels = [], []
        for img in os.listdir(path)[:n_imgs]:
            iten = os.path.join(path, img)
            itens.append(iten)
            labels.append(img)
        self.itens = itens
        self.labels = labels
    
    def __len__(self):
        return len(self.itens)
    
    def __getitem__(self, index):
        data = Image.open(self.itens[index]).convert('RGB') # retorna o tamnho da img
        data = np.asarray(torchvision.transforms.Resize(self.sizes)(data))#deixa a img no tamnho padrão
        data = np.transpose(data, (2,0,1)).astype(np.float32, copy=False)#pra botar no formato do torch (c h w)
        data = torch.from_numpy(data).div(255) #é msm coisa q data = np_array_to_tensor / 255
        return data, self.labels
    

data_path = '/content/celeba_gan/img_align_celeba'
ds = DataSet(data_path)

## DataLoader
dataloader = DataLoader(ds, batch_size=batch_size, shuffle=True)

gen = Gerador(tamnho_do_ruido).to(device)
crit = Critic().to(device)

gen_opt = torch.optim.Adam(gen.parameters(), lr=lr, betas=(0.5, 0.9))
crit_opt = torch.optim.Adam(crit.parameters(), lr=lr, betas=(0.5, 0.9))


x,y = next(iter(dataloader))
show(x)


def magic(real, fake, crit, alpha, gamma=10):
  mix_images = real * alpha + fake * (1-alpha) # 128 x 3 x 128 x 128
  mix_scores = crit(mix_images) # 128 x 1

  gradient = torch.autograd.grad(
      inputs = mix_images,
      outputs = mix_scores,
      grad_outputs=torch.ones_like(mix_scores),
      retain_graph=True,
      create_graph=True,
  )[0] # 128 x 3 x 128 x 128

  gradient = gradient.view(len(gradient), -1)   # 128 x 49152
  gradient_norm = gradient.norm(2, dim=1) 
  gp = gamma * ((gradient_norm-1)**2).mean()

  return gp


#checkpoints


pathh = 'content/'

def save_checkpoint(name):
  torch.save({
      'epoch': step_atual,
      'model_state_dict': gen.state_dict(),
      'optimizer_state_dict': gen_opt.state_dict()      
  }, f"{pathh}G-{name}.pkl")

  torch.save({
      'epoch': step_atual,
      'model_state_dict': crit.state_dict(),
      'optimizer_state_dict': crit_opt.state_dict()      
  }, f"{pathh}C-{name}.pkl")
  
  print("Saved checkpoint")

def load_checkpoint(name):
  checkpoint = torch.load(f"{pathh}G-{name}.pkl")
  gen.load_state_dict(checkpoint['model_state_dict'])
  gen_opt.load_state_dict(checkpoint['optimizer_state_dict'])

  checkpoint = torch.load(f"{pathh}C-{name}.pkl")
  crit.load_state_dict(checkpoint['model_state_dict'])
  crit_opt.load_state_dict(checkpoint['optimizer_state_dict'])

  print("Loaded checkpoint")



#LOOP D TREINAMENTOOOOOOOO

for epoca in range(epocas):
    for real, _ in tqdm(dataloader): #_ seria as classes
        cur_bs = len(real)
        real = real.to(device)

        for _ in range(critic_cycles):
            crit_opt.zero_grad()
            
            noise = get_noise(cur_bs, tamnho_do_ruido)
            
            fake = gen(noise)
            
            crit_fake_pred = crit(fake.detach())#pra n atualizar os pesos do gerador
            crit_real_pred = crit(real)
            
            alpha= torch.rand(len(real),1,1,1,device=device, requires_grad=True) # 128 x 1 x 1 x 1
            
            gp = magic(real=real, fake=fake.detach(), crit=crit, alpha=alpha)
            
            crit_loss = crit_fake_pred.mean() - crit_real_pred.mean() + gp #penalidade do gradiente
            
            mean_crit_loss = crit_loss.item() / critic_cycles # pq o loss vai acumular durante os ciclos
            
            crit_loss.backward(retain_graph=True)
            
            crit_opt.step()
        critic_losses+=[mean_crit_loss]
        

        #GENERATOR
        gen_opt.zero_grad
        
        noise = get_noise(cur_bs, tamnho_do_ruido)
        fake = gen(noise)
        
        crit_fake = crit(fake)
        
        g_loss = -crit_fake.mean()
        g_loss.backward()
        
        gen_opt.step()
        
        g_losses += [g_loss.item()]
        
        
        if (step_atual % save_interval == 0 and step_atual > 0):
            print("Saving checkpoint: ", step_atual, save_interval)
            save_checkpoint("latest")

        
        if (step_atual % save_interval == 0 and step_atual > 0):
            show(fake)
            show(real)
            
            generator_mean = sum(g_losses[-save_interval:]) / save_interval
            crit_mean = sum(critic_losses[-save_interval]) / save_interval
            print(f"Epoch: {epoca}: Step {step_atual}: Generator loss: {generator_mean}, critic loss: {crit_mean}")
            
        plt.plot(
            range(len(g_losses)),
            torch.Tensor(g_losses),
            label="Generator Loss"
        )

        plt.plot(
            range(len(g_losses)),
            torch.Tensor(critic_losses),
            label="Critic Loss"
        )
        
        plt.ylim(-150,150)
        plt.legend()
        plt.show()
    
    step_atual+=1