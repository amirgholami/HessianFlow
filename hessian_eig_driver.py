from __future__ import print_function
import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

# from progressbar import *
from utils import *
from models.c1 import *

import hessianflow as hf
import hessianflow.optimizer as hf_optm

# Training settings
parser = argparse.ArgumentParser(description = 'PyTorch Example')
parser.add_argument('--name', type = str, default = 'cifar10', metavar = 'N',
                    help='dataset')
parser.add_argument('--batch-size', type = int, default = 128, metavar = 'N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type = int, default = 200, metavar = 'N',
                    help='input batch size for testing (default: 1000)')

parser.add_argument('--seed', type = int, default = 1, metavar = 'S',
                    help = 'random seed (default: 1)')
parser.add_argument('--arch', type = str, default = 'c1',
            help = 'choose the archtecure')

parser.add_argument('--eigen-type', type = str, default = 'approximate',
                    help = 'full dataset of subset')
parser.add_argument('--resume', type = str, default = './model_param.pkl',
            help = 'choose the archtecure')

args = parser.parse_args()
# set random seed to reproduce the work
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

# get dataset
train_loader, test_loader = getData(name = args.name, train_bs = args.batch_size, test_bs = args.test_batch_size)

# get model and optimizer
model_list = {
    'c1':c1_model(),
    'c2':c2_model(),
}

model = model_list[args.arch].cuda()
model = torch.nn.DataParallel(model)
#model.load_state_dict(torch.load(args.resume))
print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))

criterion = nn.CrossEntropyLoss() 


if args.eigen_type == 'full':
    # compute the eigen information based on the whole testing data set
    eigenvalue, eigenvector = hf.get_eigen_full_dataset(model, test_loader, criterion, cuda = True, maxIter = 10, tol = 1e-2)
elif args.eigen_type == 'approximate':
    # here we choose a random batch from test_loader to compute approximating eigen information
    get_data = True
    for data, target in test_loader:
        # finish the for loop otherwise there is a warning
        if get_data:
            inputs = data
            targets = target
            get_data = False

    eigenvalue, eigenvector = hf.get_eigen(model, inputs, targets, criterion, cuda = True, maxIter = 10, tol = 1e-2)

print('Eigenvalue is: %.2f' % eigenvalue)
