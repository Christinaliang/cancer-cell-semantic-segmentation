# 1. Obtain the microscopic view of the culture dish with cells from the original 4908*3264 photo;
# 2. Remove the bright/dark background as a function of radius;
# 3. Crop it to one hundred 320*320 images.
# The code here will be reorganized.

import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import imsave
import os

from dummy_utils import img_bgremove, contrast_plus


def read_img(name, filename):
    img = plt.imread(filename+name)
    return img


def img_compress(name, filename, ratio=0.5):
    from PIL import Image

    im = Image.open(filename+name)
    size = im.size
    reduced_size = int(size[0] * ratio), int(size[1] * ratio)

    im_resized = im.resize(reduced_size, Image.ANTIALIAS)
    im_resized.save(filename+name[:-3]+"jpg", "JPEG", quality=100)
    im.close()


def img_view(img):
    plt.imshow(img)
    plt.show()


def vertical_cut(img, left, right):
    cut_idx = np.arange(left, right, 1)
    img_v = np.take(img, cut_idx, axis=1)
    return img_v


def circle_cut(img, is_PS=False, zero_padding=False):
    center_x, center_y = int(img.shape[1]/2), int(img.shape[0]/2)
    rad = center_y

    if not is_PS:
        for y_i in range(center_y, img.shape[0], 1):
            x_range = int(np.around(np.sqrt(rad**2 - (y_i - center_y)**2)))
            x_left, x_right = center_x - x_range, center_x + x_range
            img[y_i][x_right:].fill(0)
            img[y_i][:x_left].fill(0)
            img[img.shape[0]-1-y_i][x_right:].fill(0)
            img[img.shape[0]-1-y_i][:x_left].fill(0)

    img = np.take(img, np.arange(0, img.shape[1], 1)[center_x-rad:center_x+rad], axis=1)
    # For zero-padding
    if zero_padding:
        img[0, :].fill(0)
        img[-1, :].fill(0)
        img[:, 0].fill(0)
        img[:, -1].fill(0)
    return img


def border_check(img):
    print(np.sum(img[0]), np.sum(img[-1]))
    print(np.sum(img[:, 0]), np.sum(img[:, -1]))


def RGBview(img):
    red, green, blue = img.copy(), img.copy(), img.copy()
    red[:, :, (1, 2)] = 0
    green[:, :, (0, 2)] = 0
    blue[:, :, (0, 1)] = 0
    img_view(red)
    img_view(green)
    img_view(blue)


def search_LR(img, threshold, is_plot=False):
    center_y = int(img.shape[0]/2)
    center_line = img[center_y]
    mean_RGB = center_line.mean(axis=1)
    length = len(mean_RGB)

    indices = np.argwhere(mean_RGB > threshold)
    idx1, idx2 = indices[0][0], indices[-1][0]
    left = mean_RGB[idx1:idx1+1000].argmin()+idx1
    right = mean_RGB[idx2-1000:idx2].argmin()+idx2-1000

    if is_plot:
        plt.plot(range(length), mean_RGB, 'b', label='{} - {} = {}'.format(right, left, right-left))
        plt.legend(loc='lower center')
        plt.show()

    return left, right


def preprocessing(target_dir, img, img_PS, name, count, is_remove, threshold):
    left, right = search_LR(img, threshold)
    img_v = vertical_cut(img, left, right)
    PS_v = vertical_cut(img_PS, left, right)

    img_circle = circle_cut(img_v)
    PS_circle = circle_cut(PS_v, is_PS=True)
    print('Final size of circle image {}: {}'.format(name[:-4], img_circle.shape))

    if is_remove:
        img_bgremove(img_circle)

    imsave(target_dir + 'series{}/{}_img_new.tif'.format(data_number, count), img_circle)
    imsave(target_dir + 'series{}/{}_PS_new.tif'.format(data_number, count), PS_circle)


def select_images(filename, target_dir, last_idx, threshold=35, is_remove=True, ignore=[]):
    last_idx = last_idx
    count = 0
    for idx1 in range(1, last_idx+1, 1):
        for idx2 in range(1, 5, 1):
            name = str(idx1)+'-'+str(idx2)+'.tif'
            if name in ignore:
                continue
            if os.path.exists(filename+'old/'+name) and os.path.exists(filename+'PS/'+name):
                img = read_img(name, filename+'old/')
                img_PS = read_img(name, filename+'PS/')
                preprocessing(img, img_PS, name, count, is_remove, threshold)
                count += 1
    for idx in range(1, 21, 1):
        name = 'nc-'+str(idx)+'.tif'
        if name in ignore:
            continue
        if os.path.exists(filename+'old/'+name) and os.path.exists(filename+'PS/'+name):
            img = read_img(name, filename+'old/')
            img_PS = read_img(name, filename+'PS/')
            preprocessing(target_dir, img, img_PS, name, count, is_remove, threshold)
            count += 1


def img_split(img, img_PS, folder, folder_PS, index=None, keepALL=True, num_splits=10):
    size = img.shape[0]
    shift = size//num_splits

    if keepALL:
        for i in range(0, num_splits, 1):
            for j in range(0, num_splits, 1):
                simg = img[i*shift:i*shift+shift, j*shift:j*shift+shift, :]
                simg_PS = img_PS[i*shift:i*shift+shift, j*shift:j*shift+shift, :]
                if index is None:
                    imsave(folder+'{}{}_ORIG.tif'.format(i, j), simg)
                    imsave(folder_PS+'{}{}_PS.tif'.format(i, j), simg_PS)
                else:
                    imsave(folder+'{}_ORIG.tif'.format(index), simg)
                    imsave(folder_PS+'{}_PS.tif'.format(index), simg_PS)
                    index += 1
    else:
        for i in range(0, num_splits, 1):
            for j in range(0, num_splits, 1):
                simg_PS = img_PS[i*shift:i*shift+shift, j*shift:j*shift+shift, :]
                if simg_PS.min() < 255:
                    simg = img[i*shift:i*shift+shift, j*shift:j*shift+shift, :]
                    if index is None:
                        imsave(folder+'{}{}_ORIG.tif'.format(i, j), simg)
                        imsave(folder_PS+'{}{}_PS.tif'.format(i, j), simg_PS)
                    else:
                        imsave(folder+'{}_ORIG.tif'.format(index), simg)
                        imsave(folder_PS+'{}_PS.tif'.format(index), simg_PS)
                        index += 1
    return index


'''
Preprocessing
'''
# # Specify datanumbers and last_idx here
# ignore = []
#
# for data_number in data_numbers:
#     # Specify filename and target_dir here
#     print('Data number is {}:'.format(data_number))
#     select_images(filename, target_dir, last_idx, threshold=80, is_remove=False, ignore=ignore)


'''
Image split
'''

# Specify data_dir and target_dir here
imgs = [file for file in os.listdir(data_dir) if file.endswith("img_new.tif")]
masks = [file for file in os.listdir(data_dir) if file.endswith("PS_new.tif")]
print('The number of tifs:', len(imgs)+len(masks))
if len(imgs) != len(masks):
    raise RuntimeError(
        'The number of images is not equal to the number of masks.')
index = 12787
for img_file in imgs:
    img = read_img(img_file, data_dir)
    img_PS = read_img(img_file.replace('img', 'PS'), data_dir)
    print(img_file, img.shape, img_PS.shape)
    index = img_split(img, img_PS, target_dir, target_dir, index, keepALL=False)
    print('Image {} splitting finished!'.format(img_file))
print(index)
