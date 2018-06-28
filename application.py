import time
import argparse
import os

import sys
sys.path.append('/public/source/home/user_t2/CellSeg/models/')

from models import *
from celldataset import CellImages
from utils import split_train_valid_test, split_cross_validation, reinitialize_net
from utils import train, test, predict, save_epoch_stats, img_split
# from utils import get_mean, show_img_and_mask
from augmentation import *

import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.optim as optim
# from torch.backends import cudnn

import numpy as np
import matplotlib.pyplot as plt
import math


def main():
    is_resume = True

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Device is {}!'.format(device))
    best_acc = 0.  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    # Hyperparameters
    batch_size = 11
    image_size = 320
    print('Batch size: {}'.format(batch_size))

    # Model
    print('==> Building model..')
    net = DeConvNet()
    net = net.to(device)

    # # Enabling cudnn, which costs 2GB extra memory
    # if device == 'cuda':
    #     cudnn.benchmark = True
    #     print('cudnn benchmark enabled!')

    if is_resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load(
            './checkpoint/opt_methodSGD_momentum-lr0.01-momentum0.9-weight_decay0.001.t7')
        net.load_state_dict(checkpoint['net'])
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch']
        print('Start Epoch is {}'.format(start_epoch))

    img_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.6672,  0.5865,  0.5985), (1.0, 1.0, 1.0)),
    ])

    img_files = [file for file in os.listdir('./photo/') if file.endswith(".tif")]

    # Prediction
    print('==> Prediction begins..')
    net.eval()
    with torch.no_grad():
        for img_file in img_files:
            start_time = time.time()
            print('{} begins'.format(img_file))
            img = plt.imread('./photo/{}'.format(img_file))
            img_split(img, cut_size=image_size, is_overlap=False)
            mask = predict(img_file, device, net, img_transform=img_transform,
                           batch_size=batch_size, image_size=image_size)
            print('The mask of {} was predicted and saved!'.format(img_file[:-9]))
            print("--- %s seconds ---" % (time.time() - start_time))



# For Windows the conditional statement below is required
if __name__ == '__main__':
    main()
