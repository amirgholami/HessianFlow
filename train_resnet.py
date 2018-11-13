from __future__ import print_function
import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from models.resnet import resnet
from utils import *

import hessianflow as hf
import hessianflow.optimizer.optm_utils as hf_optm_utils
import hessianflow.optimizer.progressbar as hf_optm_pgb


# Training settings
parser = argparse.ArgumentParser(description='PyTorch Example')
parser.add_argument('--name', type = str, default = 'cifar10', metavar = 'N',
                    help = 'dataset')
parser.add_argument('--batch-size', type = int, default = 128, metavar = 'B',
                    help = 'input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=200, metavar='TBS',
                    help = 'input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type = int, default = 160, metavar = 'E',
                    help = 'number of epochs to train (default: 10)')

parser.add_argument('--lr', type = float, default = 0.1, metavar = 'LR',
                    help = 'learning rate (default: 0.01)')
parser.add_argument('--lr-decay', type = float, default = 0.1,
                    help = 'learning rate ratio')
parser.add_argument('--lr-decay-epoch', type = int, nargs = '+', default = [60,120],
                        help = 'Decrease learning rate at these epochs.')


parser.add_argument('--seed', type = int, default = 1, metavar = 'S',
                    help = 'random seed (default: 1)')
parser.add_argument('--weight-decay', '--wd', default = 5e-4, type = float,
                    metavar = 'W', help = 'weight decay (default: 5e-4)')
parser.add_argument('--large-ratio', type = int, default = 1,
                    help = 'large ratio')
parser.add_argument('--depth', type = int, default = 20,
            help = 'choose the depth of resnet')


parser.add_argument('--eigen-type', type = str, default = 'approximate',
                    help = 'full dataset of subset')

args = parser.parse_args()
args.cuda = True
# set random seed to reproduce the work
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)


# get dataset
train_loader, test_loader = getData(name = args.name, train_bs = args.batch_size, test_bs = args.test_batch_size)

# get model and optimizer

model = resnet(depth=args.depth).cuda()
model = torch.nn.DataParallel(model)

criterion = nn.CrossEntropyLoss() 
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)

 
inner_loop = 0
num_updates = 0
# assert that shuffle is set for train_loader
# assert and explain large ratio 
# assert that the train_loader is always set with a small batch size if not print error/warning telling
# the user to instead use large_ratio
for epoch in range(1, args.epochs + 1):
    print('\nCurrent Epoch: ', epoch)
    print('\nTraining')
    train_loss = 0.
    total_num = 0.
    correct = 0.

    for batch_idx, (data, target) in enumerate(train_loader):
        if target.size(0) < 128:
            continue
        model.train()
        # gather input and target for large batch training        
        inner_loop += 1
        # get small model update
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        output = model(data)
        loss = criterion(output, target)/float(args.large_ratio)
        loss.backward()
        train_loss += loss.item()*target.size(0)*float(args.large_ratio)
        total_num += target.size(0)
        _, predicted = output.max(1)
        correct += predicted.eq(target).sum().item()

        if inner_loop % args.large_ratio  == 0:
            num_updates += 1
            optimizer.step()
            inner_loop = 0
            optimizer.zero_grad()

        hf_optm_pgb.progress_bar(batch_idx, len(train_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss / total_num,
                        100. * correct / total_num, correct, total_num))

    
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
    print('\nCurrent Eigenvalue based on Test Dataset: %0.2f' % eigenvalue)

    if epoch in args.lr_decay_epoch:
        exp_lr_scheduler(optimizer, decay_ratio=args.lr_decay)
    
    hf_optm_utils.test(model, test_loader)     
