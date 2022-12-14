import torch
import torch.nn as nn
import torch.nn.functional as F
from .resblocks import Block

class ResNetGenerator(nn.Module):
    def __init__(self, ch=64, dim_z=128, bottom_width=4, activation=F.relu, \
            n_classes=0, distribution="normal"):
        super(ResNetGenerator, self).__init__()
        self.bottom_width = bottom_width
        self.activation = activation
        self.distribution = distribution
        self.dim_z = dim_z
        self.n_classes = n_classes
        self.l1 = nn.Linear(dim_z, (bottom_width ** 2) * ch * 16)
        nn.init.xavier_uniform_(self.l1.weight, 1.0)
        self.block2 = Block(ch * 16, ch * 8, activation=activation, upsample=True, n_classes=n_classes)
        self.block3 = Block(ch * 8, ch * 4, activation=activation, upsample=True, n_classes=n_classes)
        self.block4 = Block(ch * 4, ch * 2, activation=activation, upsample=True, n_classes=n_classes)
        self.block5 = Block(ch * 2, ch * 1, activation=activation, upsample=False, n_classes=n_classes)
        self.b6 = nn.BatchNorm2d(ch)
        nn.init.constant_(self.b6.weight, 1.0) #XXX this is different from default initialization method
        self.l6 = nn.Conv2d(ch, 1, kernel_size=3, stride=1, padding=1)
        nn.init.xavier_uniform_(self.l6.weight, 1.0)

    def forward(self, z, y):
        h = z
        h = self.l1(h)
        h = h.view(h.size(0), -1, self.bottom_width, self.bottom_width)
        h = self.block2(h, y)
        h = self.block3(h, y)
        h = self.block4(h, y)
        h = self.block5(h, y)
        h = self.b6(h)
        h = self.activation(h)
        h = self.l6(h)
        h = F.tanh(h)
        return h

