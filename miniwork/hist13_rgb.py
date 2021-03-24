import cv2
import numpy as np
import matplotlib
import os
from matplotlib import pyplot as plt
from tqdm import tqdm
from natsort import natsorted
import pandas as pd
import glob
import sys
matplotlib.use('Agg')

hsv = ('h', 's', 'v')
color = ('b','g','r')
# color2 = [
#     [0, 0, 127],
#     [0, 127, 0],
#     [127, 0, 0],
# ]
color2 = ('c', 'm', 'y')
ave_histr = np.zeros([256,3])
ave_histr_inv = np.zeros([256, 3])
black_folder = 'black_only'
white_folder = 'white_only'
double_folder = 'double'
os.makedirs(black_folder, exist_ok=True)
os.makedirs(white_folder, exist_ok=True)
os.makedirs(double_folder, exist_ok=True)
csv_black_path_list = glob.glob('?????_black_only.csv')
csv_white_path_list = glob.glob('?????_white_only.csv')
csv_black_path_list = natsorted(csv_black_path_list)
csv_white_path_list = natsorted(csv_white_path_list)
ave_white_only = pd.read_csv('ave_white_only.csv', index_col=0).values
ave_black_only = pd.read_csv('ave_black_only.csv', index_col=0).values

pbar = tqdm(total=len(csv_black_path_list))
# for path_i, path in enumerate(img_path):
for path_i in range(len(csv_black_path_list)):
    black_path = csv_black_path_list[path_i]
    white_path = csv_white_path_list[path_i]
    fname = os.path.splitext(os.path.basename(black_path))[0][:5]
    one_histr = pd.read_csv(white_path, index_col=0).values
    one_histr_inv = pd.read_csv(black_path, index_col=0).values

    for col_i, col in enumerate(color):
        plt.plot(one_histr[:, col_i], color=col, label=hsv[col_i])
        ave_histr[:, col_i] += one_histr[:, col_i]
    plt.xlim([-2, 256+2])
    plt.ylim([0, 30000000])
    plt.title(white_folder + '/' + fname + '_white_only')
    plt.savefig(white_folder + '/' + fname + '_white_only.png')
    plt.figure()

    for col_i, col in enumerate(color):
        plt.plot(one_histr_inv[:, col_i], color=col, label=hsv[col_i])
        ave_histr_inv[:, col_i] += one_histr_inv[:, col_i]
    plt.xlim([-2, 256+2])
    plt.ylim([0, 7500000])
    plt.title(black_folder + '/' + fname + '_black_only')
    plt.savefig(black_folder + '/' + fname + '_black_only.png')
    plt.figure()

    for col_i, col in enumerate(color):
        plt.plot(one_histr[:, col_i], color=col, linestyle='dashed', label=col + ' white')
        plt.plot(one_histr_inv[:, col_i], color=col, label=col + ' black')
        ave_histr_inv[:, col_i] += one_histr_inv[:, col_i]
    plt.xlim([-2, 256+2])
    plt.ylim([0, 30000000])
    plt.title(double_folder + '/' + fname + '_double')
    plt.legend(loc='upper right')
    plt.savefig(double_folder + '/' + fname + '_double.png')
    # plt.legend(bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=1, fontsize=18)
    plt.figure()

    pbar.update(1)  # プロセスバーを進行
pbar.close()  # プロセスバーの終了

for col_i, col in enumerate(color):
    plt.plot(ave_white_only[:, col_i], color=col)
    ave_histr[:, col_i] += ave_white_only[:, col_i]
plt.xlim([-2, 256 + 2])
plt.ylim([0, 30000000])
plt.title(white_folder + '/' + 'ave_white_only')
plt.savefig(white_folder + '/' + 'ave_white_only.png')
plt.figure()

for col_i, col in enumerate(color):
    plt.plot(ave_black_only[:, col_i], color=col)
    ave_histr_inv[:, col_i] += ave_black_only[:, col_i]
plt.xlim([-2, 256 + 2])
plt.ylim([0, 7500000])
plt.title(black_folder + '/' + 'ave_black_only')
plt.savefig(black_folder + '/' + 'ave_black_only.png')
plt.figure()
pbar.update(1)  # プロセスバーを進行