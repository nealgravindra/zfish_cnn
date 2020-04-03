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

class cnn(nn.Module):
    def __init__(self):
        super(cnn, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1,
                               out_channels=4,
                               kernel_size=5,
                               stride=1,
                               padding=0)

        self.pool = nn.MaxPool2d(kernel_size=2,
                                 stride=2,
                                 padding=0)

        self.pool3 = nn.MaxPool2d(kernel_size=1,
                                  stride=2,
                                  padding=0)

        self.conv2 = nn.Conv2d(in_channels=4,
                               out_channels=8,
                               kernel_size=5,
                               stride=1,
                               padding=0)

        self.conv3 = nn.Conv2d(in_channels=8,
                               out_channels=16,
                               kernel_size=5,
                               stride=1,
                               padding=0)

        self.fc1 = nn.Linear(in_features=16 * 13 * 13,
                             out_features=100,
                             bias=True)

        self.fc2 = nn.Linear(in_features=100,
                             out_features=25,
                             bias=True)

        self.fc3 = nn.Linear(in_features=25,
                             out_features=1, # number of classes
                             bias=True)

        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.pool(F.leaky_relu(self.conv1(x))) # 128 --> 124 --> 62
        x = self.pool(F.leaky_relu(self.conv2(x))) # 62 --> 58 --> 29
        x = self.pool3(F.leaky_relu(self.conv3(x))) # 29 --> 25 --> 13
        x = x.view(-1, 16 * 13 * 13) # reshape
        x = F.leaky_relu(self.fc1(x))
        if True :
            x = self.dropout(x)
        x = F.leaky_relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        x = x.squeeze() # kick out extra dim
        return x
