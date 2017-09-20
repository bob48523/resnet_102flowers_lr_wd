#!/usr/bin/env python3

import argparse
import torch

import torch.nn as nn
import torch.optim as optim

import torch.nn.functional as F
from torch.autograd import Variable
import torch.backends.cudnn as cudnn

import torchvision.datasets as dset
import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader

import os
import sys
import math

import shutil

import setproctitle

import resnet

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchSz', type=int, default=128)
    parser.add_argument('--nEpochs', type=int, default=300)
    parser.add_argument('--no-cuda', action='store_true')
    parser.add_argument('--save')
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--wd', type=float, default=0.0005)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--opt', type=str, default='sgd',
                        choices=('sgd', 'adam', 'rmsprop'))
    args = parser.parse_args()

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    args.save = args.save or 'work/wd/weightdecay0.001'
    setproctitle.setproctitle(args.save)

    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    if os.path.exists(args.save):
        shutil.rmtree(args.save)
    os.makedirs(args.save, exist_ok=True)

    trainLoader = torch.utils.data.DataLoader(
        dset.ImageFolder('/dataset/102flower/102flowers_20_10/train/', transforms.Compose([
            transforms.Scale(72),
            transforms.RandomSizedCrop(64),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.434, 0.385, 0.296],
                                     std=[0.285, 0.238, 0.262])
        ])),
        batch_size=args.batchSz, shuffle=True,
        num_workers=2, pin_memory=True)

    valLoader = torch.utils.data.DataLoader(
        dset.ImageFolder('/dataset/102flower/102flowers_20_10/val/', transforms.Compose([
            transforms.Scale(72),
            transforms.CenterCrop(64),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.434, 0.385, 0.296],
                                     std=[0.285, 0.238, 0.262])
        ])),
        batch_size=args.batchSz, shuffle=False,
        num_workers=2, pin_memory=True)

    net = resnet.ResNet18()
    print('  + Number of params: {}'.format(
        sum([p.data.nelement() for p in net.parameters()])))
    if args.cuda:
        net = net.cuda()
        net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
        cudnn.benchmark = True
    
    if args.opt == 'sgd':
        optimizer = optim.SGD(net.parameters(), lr=args.lr,
                            momentum=0.9, weight_decay=args.wd)
    elif args.opt == 'adam':
        optimizer = optim.Adam(net.parameters(), weight_decay=1e-4)
    elif args.opt == 'rmsprop':
        optimizer = optim.RMSprop(net.parameters(), weight_decay=1e-4)

    trainF = open(os.path.join(args.save, 'train.csv'), 'w')
    testF = open(os.path.join(args.save, 'test.csv'), 'w')

    for epoch in range(1, args.nEpochs + 1):
        adjust_opt(args.opt, optimizer, args.lr, epoch)
        train(args, epoch, net, trainLoader, optimizer, trainF)
        test(args, epoch, net, valLoader, optimizer, testF)
        torch.save(net, os.path.join(args.save, 'latest.pth'))
        os.system('./plot.py {} &'.format(args.save))

    trainF.close()
    testF.close()

def train(args, epoch, net, trainLoader, optimizer, trainF):
    net.train()
    nProcessed = 0
    nTrain = len(trainLoader.dataset)
    for batch_idx, (data, target) in enumerate(trainLoader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = net(data)
        #print(output)
        criterion = nn.CrossEntropyLoss()
        loss = criterion(output, target)
        #loss = F.nll_loss(output, target)
        # make_graph.save('/tmp/t.dot', loss.creator); assert(False)
        loss.backward()
        optimizer.step()
        nProcessed += len(data)
        pred = output.data.max(1)[1] # get the index of the max log-probability
        incorrect = pred.ne(target.data).cpu().sum()
        err = 100.*incorrect/len(data)
        partialEpoch = epoch + batch_idx / len(trainLoader) - 1
        print('Train Epoch: {:.2f} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tError: {:.6f}'.format(
            partialEpoch, nProcessed, nTrain, 100. * batch_idx / len(trainLoader),
            loss.data[0], err))

        trainF.write('{},{},{}\n'.format(partialEpoch, loss.data[0], err))
        trainF.flush()
    

def test(args, epoch, net, testLoader, optimizer, testF):
    net.eval()
    test_loss = 0
    incorrect = 0
    for data, target in testLoader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = net(data)
        criterion = nn.CrossEntropyLoss()
        test_loss += criterion(output, target)
        #test_loss += F.nll_loss(output, target).data[0]
        pred = output.data.max(1)[1] # get the index of the max log-probability
        incorrect += pred.ne(target.data).cpu().sum()

    test_loss = test_loss
    test_loss /= len(testLoader) # loss function already averages over batch size
    nTotal = len(testLoader.dataset)
    err = 100.*incorrect/len(testLoader.dataset)
    print('\nTest set: Average loss: {:.4f}, Error: {}/{} ({:.0f}%)\n'.format(
        test_loss.data[0], incorrect, nTotal, err))

    testF.write('{},{},{}\n'.format(epoch, test_loss.data[0], err))
    testF.flush()

def adjust_opt(optAlg, optimizer, base_lr, epoch):
    if optAlg == 'sgd':
        if epoch < 150: lr = base_lr
        elif epoch == 150: lr = base_lr*0.1
        elif epoch == 200: lr = base_lr*0.01
        else: return

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

if __name__=='__main__':
    main()
