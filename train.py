#*
# @file ABSA training driver based on arxiv:1810.01021 
# Copyright (c) Zhewei Yao, Amir Gholami
# All rights reserved.
# This file is part of HessianFlow library.
#
# HessianFlow is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# HessianFlow is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with HessianFlow.  If not, see <http://www.gnu.org/licenses/>.
#*
from __future__ import print_function
import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from models.c1 import *
from models.resnet import *
from utils import *

import hessianflow as hf
import hessianflow.optimizer as hf_optm

# Training settings
parser = argparse.ArgumentParser(description='PyTorch Example')
parser.add_argument('--name', type = str, default = 'cifar10', metavar = 'N',
                    help = 'dataset')
parser.add_argument('--batch-size', type = int, default = 128, metavar = 'B',
                    help = 'input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=200, metavar='TBS',
                    help = 'input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type = int, default = 90, metavar = 'E',
                    help = 'number of epochs to train (default: 10)')

parser.add_argument('--lr', type = float, default = 0.1, metavar = 'LR',
                    help = 'learning rate (default: 0.01)')
parser.add_argument('--lr-decay', type = float, default = 0.2,
                    help = 'learning rate ratio')
parser.add_argument('--lr-decay-epoch', type = int, nargs = '+', default = [30, 60, 90],
                        help = 'Decrease learning rate at these epochs.')


parser.add_argument('--seed', type = int, default = 1, metavar = 'S',
                    help = 'random seed (default: 1)')
parser.add_argument('--arch', type = str, default = 'ResNet',
            help = 'choose the archtecure')
parser.add_argument('--large-ratio', type = int, default = 128,
                    help = 'large ratio')
parser.add_argument('--depth', type = int, default = 20,
            help = 'choose the depth of resnet')

parser.add_argument('--method', type = str, default = 'absa',
            help = 'choose the method to train you model')

args = parser.parse_args()
# set random seed to reproduce the work
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)


# get dataset
train_loader, test_loader = getData(name = args.name, train_bs = args.batch_size, test_bs = args.test_batch_size)

transform_train = transforms.Compose([
        transforms.ToTensor(),
    ])

trainset = datasets.CIFAR10(root='../data', train = True, download = True, transform = transform_train)
hessian_loader = torch.utils.data.DataLoader(trainset, batch_size = 128, shuffle = True)


# get model and optimizer
model_list = {
    'c1':c1_model(),
    'ResNet': resnet(depth = args.depth),
}

model = model_list[args.arch].cuda()
model = torch.nn.DataParallel(model)

criterion = nn.CrossEntropyLoss() 
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)

########### training  
if args.method == 'absa':
    model, num_updates=hf_optm.absa(model, train_loader, hessian_loader, test_loader, criterion, optimizer, args.epochs, args.lr_decay_epoch, args.lr_decay,
        batch_size = args.batch_size, max_large_ratio = args.large_ratio, adv_ratio = 0.2, eps = 0.005, cuda = True, print_flag = True)
elif args.method == 'sgd':
    model, num_updates = hf_optm.baseline(model, train_loader, test_loader, criterion, optimizer, args.epochs, args.lr_decay_epoch, 
            args.lr_decay, batch_size = args.batch_size, max_large_ratio = args.large_ratio, cuda = True)
