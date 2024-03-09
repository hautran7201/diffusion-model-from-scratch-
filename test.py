import torch 
from sd.attention import *
from sd.diffusion import Unet


model = Unet()
x = torch.rand((1, 3, 3, 3))
context = torch.rand((1, 77, 768))
time = torch.rand((1, 320))

print(model(x, context, time))