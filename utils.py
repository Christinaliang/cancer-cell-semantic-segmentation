import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import imsave

import torch
from torch.utils.data import DataLoader


def split_train_test(data_dir, test_size=2000):
    tifs = [file for file in os.listdir(data_dir) if (
        file.endswith("ORIG.tif") or file.endswith("PS.tif"))]
    size = len(tifs)//2
    test_indices = np.random.choice(range(size), test_size, replace=False)
    train_indices = np.delete(range(size), test_indices)
    np.random.shuffle(train_indices)

    train_indices, test_indices = list(map(str, train_indices)), list(map(str, test_indices))
    return train_indices, test_indices


def split_cross_validation(data_dir, splits=8):
    tifs = [file for file in os.listdir(data_dir) if (
        file.endswith("ORIG.tif") or file.endswith("PS.tif"))]
    size = len(tifs)//2
    all_indices = np.arange(size)
    np.random.shuffle(all_indices)

    test_size = int(size/splits)
    cv_indices = []
    for i in range(splits):
        test_indices = all_indices[i*test_size:(i+1)*test_size]
        train_indices = np.delete(all_indices, np.s_[i*test_size:(i+1)*test_size])
        train_indices, test_indices = list(map(str, train_indices)), list(map(str, test_indices))
        cv_indices.append((train_indices, test_indices))
    return cv_indices


def get_mean(dataset):
    '''Compute the mean value of dataset.'''
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    mean = torch.zeros(3)
    print('==> Computing mean..')
    for idx, sample in enumerate(dataloader):
        if idx % 100 == 0:
            print('{} samples finished calculation.'.format(idx))
        input = sample[0]
        for i in range(3):
            mean[i] += input[:, i, :, :].mean()
    mean.div_(len(dataset))
    return mean


def show_img_and_mask(img, mask):
    img[0].add_(0.6672)
    img[1].add_(0.5865)
    img[2].add_(0.5985)
    npimg, npmask = img.numpy(), mask.numpy()
    img = (np.transpose(npimg, (1, 2, 0))*255).astype(np.uint8)
    mask = np.transpose(npmask, (1, 2, 0))
    both = np.concatenate((img, mask), axis=1)
    plt.imshow(both)
    plt.show()


def progress_bar(msg):
    pass


def reinitialize_net(net):
    count = 0
    for child in net.children():
        if hasattr(child, 'reset_parameters'):
            child.reset_parameters()
            count += 1
        elif isinstance(child, torch.nn.Sequential):
            for ly in child:
                if hasattr(ly, 'reset_parameters'):
                    ly.reset_parameters()
                    count += 1
    print('{} layers have been initialized!'.format(count))


def save_epoch_results(epoch, train_results, test_results, hps):
    if not os.path.isdir('epoch_results'):
        os.mkdir('epoch_results')

    name = '-'.join([hp+str(value) for hp, value in hps.items()])+'.txt'
    with open('./epoch_results/'+name, 'a') as resfile:
        resfile.write('{0:} {1:.4f} {2:.2f} {3:.2f} {4:.4f} {5:.2f} {6:.2f}'.format(
            epoch, train_results[0], train_results[1], train_results[2], test_results[0], test_results[1], test_results[2]))
        resfile.write('\n')
        

def train(epoch, device, trainloader, net, criterion, optimizer, image_size, is_print_mb=True):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    intersect_area, union_area = 0, 0

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        # Check predicted is a variable in the graph or not
        train_loss += (loss.item()*targets.size(0))
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += (predicted.eq(targets).sum().item()/(image_size**2))

        intersect = predicted*targets
        union = predicted+targets-intersect
        intersect_area += intersect.sum().item()
        union_area += union.sum().item()

        if is_print_mb and batch_idx % 100 == 0:
            if union_area > 0:
                print('minibatch: {0:3};   cur_Loss: {1:.4f};   cur_Acc: {2:.2f};   IOU: {3:.2f}'.format(
                    batch_idx, train_loss/total, 100.*correct/total, 100.*intersect_area/union_area))
            else:
                print('minibatch: {0:3};   cur_Loss: {1:.4f};   cur_Acc: {2:.2f};   IOU: TBD'.format(
                    batch_idx, train_loss/total, 100.*correct/total))
    print('Training finished. Loss: {0:.4f};   Acc: {1:.2f};   IOU: {2:.2f}'.format(
        train_loss/total, 100.*correct/total, 100.*intersect_area/union_area))
    return train_loss/total, 100.*correct/total, 100.*intersect_area/union_area


def test(epoch, device, testloader, net, criterion, image_size, best_acc, hps, is_savenet=True, is_print_mb=True, is_savepred=False):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    intersect_area, union_area = 0, 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += (loss.item()*targets.size(0))
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += (predicted.eq(targets).sum().item()/(image_size**2))

            intersect = predicted*targets
            union = predicted+targets-intersect
            intersect_area += intersect.sum().item()
            union_area += union.sum().item()

            if is_savepred:
                if not os.path.isdir('prediction'):
                    os.mkdir('prediction')
                prediction = predicted.cpu().numpy()
                for i in range(len(prediction)):
                    mask_pred = prediction[i].astype(np.uint8)
                    imsave('./prediction/{}_{}.tif'.format(batch_idx, i), (1-mask_pred)*255)

            if is_print_mb and batch_idx % 100 == 0:
                if union_area > 0:
                    print('minibatch: {0:3};   cur_Loss: {1:.4f};   cur_Acc: {2:.2f};   IOU: {3:.2f}'.format(
                        batch_idx, test_loss/total, 100.*correct/total, 100.*intersect_area/union_area))
                else:
                    print('minibatch: {0:3};   cur_Loss: {1:.4f};   cur_Acc: {2:.2f};   IOU: TBD'.format(
                        batch_idx, test_loss/total, 100.*correct/total))
            # progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            #              % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
        print('Testing  finished. Loss: {0:.4f};   Acc: {1:.2f};   IOU: {2:.2f}'.format(
            test_loss/total, 100.*correct/total, 100.*intersect_area/union_area))

    # Save checkpoint.
    acc = correct/total
    if acc > best_acc:
        if is_savenet:
            print('Saving..')
            state = {
                'net': net.state_dict(),
                'acc': acc,
                'epoch': epoch,
            }
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            name = '-'.join([hp+str(value) for hp, value in hps.items()])+'.t7'
            torch.save(state, './checkpoint/'+name)
        best_acc = acc
    return test_loss/total, 100.*correct/total, 100.*intersect_area/union_area, best_acc

