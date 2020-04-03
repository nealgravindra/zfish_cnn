import tifffile
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision as tv
from torch.autograd import Variable
from torchvision.utils import save_image
import torch.optim as optim
from torch.utils import data

# model

class ConvDenoiser(nn.Module):
    def __init__(self,latent_size):
        super(ConvDenoiser, self).__init__()

        self.latent_size=latent_size

        # encoder layers
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 16, 3, padding=1)
        self.conv3 = nn.Conv2d(16, 8, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2, return_indices=True)

        # extract latent dimension
        self.efc1 = nn.Linear(8*16*16,latent_size)
        self.dfc1 = nn.Linear(latent_size,8*16*16)

        # decoder layers
        self.unpool = nn.MaxUnpool2d(2,2)
        self.t_conv1 = nn.ConvTranspose2d(8, 16, 3, padding=1)  # kernel_size=3 to get to a 7x7 image output
        self.t_conv2 = nn.ConvTranspose2d(16, 32, 3, padding=1)
        self.t_conv3 = nn.ConvTranspose2d(32, 1, 3, padding=1)


    def encoder(self, x):
        x = F.leaky_relu(self.conv1(x))
        x,idx1 = self.pool(x)
        x = F.leaky_relu(self.conv2(x))
        x,idx2 = self.pool(x)
        x = torch.tanh(self.conv3(x))
        x,idx3 = self.pool(x)
        x = x.view(-1,8*16*16)
        x = self.efc1(x)
        return x, idx1, idx2, idx3


    def decoder(self,x, idx1, idx2, idx3):
        x = self.dfc1(x)
        x = x.view(-1,8,16,16)
        x = self.unpool(x,idx3)
        x = F.leaky_relu(self.t_conv1(x))
        x = self.unpool(x,idx2)
        x = F.leaky_relu(self.t_conv2(x))
        x = self.unpool(x,idx1)
        x = torch.tanh(self.t_conv3(x))
        return x

    def forward(self,x):
        z,idx1,idx2,idx3 = self.encoder(x)
        decoded = self.decoder(z,idx1,idx2,idx3)
        return z,decoded
