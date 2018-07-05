import os
import argparse

import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Area calculation')
parser.add_argument('--day', '-d', default=0, type=int,
                    help='Input the date as int, such as 0 or 7.')
args = parser.parse_args()


dir = './mask/'
treat_files = [file for file in os.listdir(dir) if file[:2]!='NC' and file[:2]!='nc' and file.endswith('_PRED.tif')]
nc_files = [file for file in os.listdir(dir) if (file[:2]=='NC' or file[:2]=='nc') and file.endswith('_PRED.tif')]

mask_files=sorted(treat_files, key = lambda x: int(x[:x.find('-')]))+nc_files
day = args.day

with open(dir+'areas{}.txt'.format(day), 'a') as area_file:
	for mask_file in mask_files:
		mask = plt.imread(dir+mask_file)
		zeros = mask==0
		count=zeros.sum()
		print('{0:>13}: {1:7} pixels found, {2:.2f}%'.format(mask_file, count, 100*count/(3200*3520)))
		area_file.write('{}, {}\n'.format(mask_file[:-9]+mask_file[-4:], count))
