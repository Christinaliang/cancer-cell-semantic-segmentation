# 1. Assuming the bright/dark backgound is a function of dish radius, fit it with a quartic function.
# 2. Compute the mean background and substract the mean from each pixel in a radius-wise way.
# The code here will be reorganized.

import numpy as np
import matplotlib.pyplot as plt


def get_circleRGB(img, rad):
    '''
    The image is a sqaure matrix.
    '''
    def get_peri(img, rad):
        center_x, center_y = int(img.shape[1]/2), int(img.shape[0]/2)
        peri = []
        for y_i in range(center_y, center_y + rad, 1):
            x_range = int(np.around(np.sqrt(rad**2 - (y_i - center_y)**2)))
            x_left, x_right = center_x - x_range, center_x + x_range
            peri.append(img[y_i][x_right])
            peri.append(img[y_i][x_left])
            peri.append(img[2*center_y-y_i][x_right])
            peri.append(img[2*center_y-y_i][x_left])
        return np.array(peri)
    peri1 = get_peri(img, rad)
    img_tp = np.transpose(img, (1, 0, 2))
    peri2 = get_peri(img_tp, rad)
    circle_RGB = (peri1.mean(axis=0)+peri2.mean(axis=0))/2.
    circle_max = np.maximum(peri1.max(axis=0), peri2.max(axis=0))
    circle_min = np.minimum(peri1.min(axis=0), peri2.min(axis=0))
    return circle_RGB, circle_max, circle_min


def get_border(img):
    '''
    The image is a sqaure matrix.
    '''
    borders = [None]*img.shape[0]
    center_x, center_y = int(img.shape[1]/2), int(img.shape[0]/2)
    rad = center_y
    for y_i in range(center_y, img.shape[0], 1):
        x_range = int(np.around(np.sqrt(rad**2 - (y_i - center_y)**2)))
        x_left, x_right = center_x - x_range, center_x + x_range
        borders[y_i] = (x_left, x_right)
        borders[img.shape[0]-1-y_i] = (x_left, x_right)
    return borders


def plot_bgcurve(rads, circle_RGBs,  circle_maxs, circle_mins, img_circle, step):
    plt.figure()

    plt.plot(rads, circle_RGBs[:, 0], 'r', label='Average RED')
    plt.plot(rads, circle_RGBs[:, 1], 'g', label='Average GREEN')
    plt.plot(rads, circle_RGBs[:, 2], 'b', label='Average BLUE')

    plt.plot(rads, circle_maxs[:, 0], 'ro', label='Max RED')
    plt.plot(rads, circle_maxs[:, 1], 'go', label='Max GREEN')
    plt.plot(rads, circle_maxs[:, 2], 'bo', label='Max BLUE')

    plt.plot(rads, circle_mins[:, 0], 'rs', label='Min RED')
    plt.plot(rads, circle_mins[:, 1], 'gs', label='Min GREEN')
    plt.plot(rads, circle_mins[:, 2], 'bs', label='Min BLUE')

    plt.xlim(0., img_circle.shape[0]/2)
    plt.ylim(0, 255)
    plt.xlabel('Radius (step={})'.format(step))
    plt.ylabel('RGB values (0~255)')
    plt.legend(loc='lower left')
    plt.show()


def fit_bgcurve(rads, size, circle_RGBs, degree=4, is_plot=False, size_keep=500, factor=0.9, is_whitecenter=False):
    '''
    Fit RGB background curves with a quartic function.
    Return discrete enhancing RGB functions. The enhancement is multiplied by 0.9 factor.
    In the white center scenario, RGB values with radius in [0:500] remain unchanged.
    '''
    y_RED, y_GREEN, y_BLUE = circle_RGBs[:, 0], circle_RGBs[:, 1], circle_RGBs[:, 2]

    coefs_RED, coefs_GREEN, coefs_BLUE = np.polyfit(rads, y_RED, degree), np.polyfit(rads, y_GREEN, degree), np.polyfit(rads, y_BLUE, degree)
    x_new = np.arange(0, int(size/2)+10, 1)
    r_new, g_new, b_new = np.polyval(coefs_RED, x_new), np.polyval(coefs_GREEN, x_new), np.polyval(coefs_BLUE, x_new)

    if is_plot:
        plt.figure()
        plt.plot(rads, y_RED,  'ro', markersize=4)
        plt.plot(rads, y_GREEN,  'go', markersize=4)
        plt.plot(rads, y_BLUE,  'bo', markersize=4)
        plt.plot(x_new, r_new, 'r')
        plt.plot(x_new, g_new, 'g')
        plt.plot(x_new, b_new, 'b')
        plt.show()

    if is_whitecenter:
        r_level, g_level, b_level = r_new[:size_keep].mean(), g_new[:size_keep].mean(), b_new[:size_keep].mean()
        r_change, g_change, b_change = r_level-r_new, g_level-g_new, b_level-b_new
        r_change[:size_keep] = np.zeros(size_keep)
        g_change[:size_keep] = np.zeros(size_keep)
        b_change[:size_keep] = np.zeros(size_keep)
        return np.around(r_change*factor).astype(np.uint8), np.around(g_change*factor).astype(np.uint8), np.around(b_change*factor).astype(np.uint8)

    else:
        r_level, g_level, b_level = y_RED.mean(), y_GREEN.mean(), y_BLUE.mean()
        r_change, g_change, b_change = r_level-r_new, g_level-g_new, b_level-b_new
        return np.around(r_change*factor).astype(int), np.around(g_change*factor).astype(int), np.around(b_change*factor).astype(int)


def BG_reduction(img, r_change, g_change, b_change):
    borders = get_border(img)

    center_x, center_y = int(img.shape[1]/2), int(img.shape[0]/2)
    for y_idx, x_idx in np.ndindex(*img.shape[:2]):
        if x_idx == 0 and y_idx % 100 == 0:
            print('{} of {} lines passed'.format(y_idx, img.shape[0]))
        left, right = borders[y_idx]
        if left <= x_idx < right:
            rad = int(np.around(np.sqrt((y_idx-center_y)**2+(x_idx-center_x)**2)))

            value = 255-img[y_idx][x_idx][0]
            if r_change[rad] > value:
                img[y_idx][x_idx][0] = 255
                print('R>255: y={}, x={}, exceed={}'.format(y_idx, x_idx, r_change[rad]-value))
            else:
                img[y_idx][x_idx][0] += r_change[rad]
                img[y_idx][x_idx][0] = img[y_idx][x_idx][0].astype(np.uint8)

            value = 255-img[y_idx][x_idx][1]
            if g_change[rad] > value:
                img[y_idx][x_idx][1] = 255
                print('G>255: y={}, x={}, exceed={}'.format(y_idx, x_idx, g_change[rad]-value))
            else:
                img[y_idx][x_idx][1] += g_change[rad]
                img[y_idx][x_idx][1] = img[y_idx][x_idx][1].astype(np.uint8)

            value = 255-img[y_idx][x_idx][2]
            if b_change[rad] > value:
                img[y_idx][x_idx][2] = 255
                print('B>255: y={}, x={}, exceed={}'.format(y_idx, x_idx, b_change[rad]-value))
            else:
                img[y_idx][x_idx][2] += b_change[rad]
                img[y_idx][x_idx][2] = img[y_idx][x_idx][2].astype(np.uint8)


def contrast_plus(img, pct=0.2):
    mean_R, mean_G, mean_B = 128., 128., 128.
    borders = get_border(img)

    for y_idx, x_idx in np.ndindex(*img.shape[:2]):
        if x_idx == 0 and y_idx % 100 == 0:
            print('{} of {} lines passed'.format(y_idx, img.shape[0]))
        left, right = borders[y_idx]
        if left <= x_idx < right:
            img[y_idx][x_idx][0] = np.uint8(mean_R + (img[y_idx][x_idx][0] - mean_R)*(1.+pct))
            img[y_idx][x_idx][1] = np.uint8(mean_G + (img[y_idx][x_idx][1] - mean_G)*(1.+pct))
            img[y_idx][x_idx][2] = np.uint8(mean_B + (img[y_idx][x_idx][2] - mean_B)*(1.+pct))


def img_bgremove(img_circle, step=10):
    rads = np.arange(1, int(img_circle.shape[0]/2), step)
    circle_RGBs = []
    circle_maxs = []
    circle_mins = []
    for rad in rads:
        res = get_circleRGB(img_circle, rad)
        circle_RGBs.append(res[0])
        circle_maxs.append(res[1])
        circle_mins.append(res[2])
    circle_RGBs = np.array(circle_RGBs)
    circle_maxs = np.array(circle_maxs)
    circle_mins = np.array(circle_mins)

    # plot_bgcurve(rads, circle_RGBs, circle_maxs, circle_mins, img_circle, step)

    prms = fit_bgcurve(rads, img_circle.shape[0], circle_RGBs)
    BG_reduction(img_circle, *prms)
