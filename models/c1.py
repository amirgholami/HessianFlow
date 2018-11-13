
''' Impelementation of C1 and C2 model used in arxiv:1810.01021
'''
from __future__ import print_function
import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
    
class c1_model(nn.Module):

    def __init__(self, num_classes=10):
        super(c1_model, self).__init__()
        self.conv1=nn.Conv2d(3, 64, kernel_size=5, stride=1, padding=2)# 32x32x3 -> 32x32x64
        self.bn1=nn.BatchNorm2d(64)
        self.conv2=nn.Conv2d(64, 64, kernel_size=5, stride =1, padding=2)# 16x16x64 -> 16x16x64
        self.bn2=nn.BatchNorm2d(64)
        self.fc1= nn.Linear(64*8*8, 384)
        self.fc2=nn.Linear(384,192)
        self.fc3=nn.Linear(192,num_classes)

        
    def forward(self, x):
        x = F.max_pool2d(self.bn1(F.relu(self.conv1(x))),3,2,1)
        x = F.max_pool2d(self.bn2(F.relu(self.conv2(x))),3,2,1)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
   


class c2_model(nn.Module):
    def __init__(self):
        super(c2_model, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64,128, kernel_size=3)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128,128, kernel_size=3)
        self.bn4 = nn.BatchNorm2d(128)
        self.conv_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(128*5*5, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 10)
        self.drop = nn.Dropout()

    def forward(self, x):
        x = self.bn1(F.relu(self.conv1(x)))
        x = F.max_pool2d(self.bn2(F.relu(self.conv2(x)))  ,2)
        x = self.bn3(F.relu(self.conv3(x)))
        x = F.max_pool2d(self.bn4(F.relu(self.conv4(x)))  ,2)
        #x = self.conv_drop(x)
        x = x.view(-1, 128*5*5) 
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
