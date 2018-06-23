import time
import argparse
import os

import sys
sys.path.append(dir) # add the directory to sys.path if needed

from models import *
from celldataset import CellImages
from utils import split_train_test, split_cross_validation, reinitialize_net
from utils import train, test, save_epoch_results
# from utils import get_mean, show_img_and_mask
from augmentation import *

import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.optim as optim
from torch.backends import cudnn

import numpy as np
import math


def main():
    is_resume = False

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Device is {}!'.format(device))
    best_acc = 0.  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    # Hyperparameters
    batch_size = 16
    lr = 0.01
    momentum = 0.9
    weight_decay = 1e-3
    opt_method = 'SGD_momentum'
    image_size = 320

    hps = {'opt_method': opt_method, 'lr': lr, 'momentum': momentum, 'weight_decay': weight_decay}
    print('Batch size: {}, '.format(batch_size)+', '.join([hp+': '+str(value) for hp, value in hps.items()]))

    # Model
    print('==> Building model..')
    net = DeConvNet()
    net = net.to(device)

    # Enabling cudnn, which usually costs around 2GB extra memory
    if device == 'cuda':
        cudnn.benchmark = True
        print('cudnn benchmark enabled!')
        
    if is_resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load('./checkpoint/training_saved.t7') # Load your saved model
        net.load_state_dict(checkpoint['net'])
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch']
        
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)

    # Data
    print('==> Preparing data..')
    img_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.6672,  0.5865,  0.5985), (1.0, 1.0, 1.0)),
    ])

    joint_transform = JointCompose([
        ArrayToPIL(),
        RandomRotateFlip(),
    ])
    
    data_dir = datadir # specify the data directory if needed
    train_indices, test_indices = split_train_test(data_dir, test_size=2000)

    trainset = CellImages(data_dir, train_indices, img_transform=img_transform, joint_transform=joint_transform)
    print('Trainset size: {}. Number of mini-batch: {}'.format(len(trainset), math.ceil(len(trainset)/batch_size)))
    testset = CellImages(data_dir, test_indices, img_transform=img_transform)
    print('Testset size: {}. Number of mini-batch: {}'.format(len(testset), math.ceil(len(testset)/batch_size)))

    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=8)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=8)

    # Train and Test

    print('==> Training begins..')
    for epoch in range(start_epoch, start_epoch+200):
        start_time = time.time()
        train_results = train(epoch, device, trainloader, net, criterion, optimizer, image_size, is_print_mb=False)
        test_results = test(epoch, device, testloader, net, criterion, image_size, best_acc, hps, is_save=True, is_print_mb=False)
        best_acc = test_results[-1]
        save_epoch_results(epoch, train_results, test_results, hps)
        print("--- %s seconds ---" % (time.time() - start_time))


# For Windows the conditional statement below is required
if __name__ == '__main__':
    main()
