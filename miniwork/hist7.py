import cv2
import numpy as np
import matplotlib
import os
from matplotlib import pyplot as plt
from tqdm import tqdm
from natsort import natsorted
import pandas as pd
import glob
matplotlib.use('Agg')

# 24576 * 19608
color = ('b','g','r')
ave_histr = np.zeros([256,3])
ave_histr_inv = np.zeros([256, 3])
black_folder = 'diff_black_only'
white_folder = 'diff_white_only'
os.makedirs(black_folder, exist_ok=True)
os.makedirs(white_folder, exist_ok=True)
csv_black_path_list = glob.glob('?????_black_only.csv')
csv_white_path_list = glob.glob('?????_white_only.csv')
ave_white_only = pd.read_csv('ave_white_only.csv', index_col=0).values
ave_black_only = pd.read_csv('ave_black_only.csv', index_col=0).values
ave_diff_white_only = np.zeros([256,3])
ave_diff_black_only = np.zeros([256,3])

pbar = tqdm(total=len(csv_black_path_list))
# for path_i, path in enumerate(img_path):
for path_i in range(len(csv_black_path_list)):
    black_path = csv_black_path_list[path_i]
    white_path = csv_white_path_list[path_i]
    fname = os.path.splitext(os.path.basename(black_path))[0][:5]
    one_histr = pd.read_csv(white_path, index_col=0).values - ave_white_only
    one_histr_inv = pd.read_csv(black_path, index_col=0).values - ave_black_only

    for col_i, col in enumerate(color):
        plt.plot(one_histr[:, col_i], color=col)
        ave_histr[:, col_i] += one_histr[:, col_i]
        ave_diff_white_only[:, col_i] += one_histr[:, col_i]
    plt.xlim([-2, 256+2])
    plt.ylim([-28000000, 28000000])
    plt.title(white_folder + '/' + fname + '_diff_white_only')
    plt.savefig(white_folder + '/' + fname + '_diff_white_only.png')
    plt.figure()

    for col_i, col in enumerate(color):
        plt.plot(one_histr_inv[:, col_i], color=col)
        ave_histr_inv[:, col_i] += one_histr_inv[:, col_i]
        ave_diff_black_only[:, col_i] += one_histr_inv[:, col_i]
    plt.xlim([-2, 256+2])
    plt.ylim([-7500000, 7500000])
    plt.title(black_folder + '/' + fname + '_diff_black_only')
    plt.savefig(black_folder + '/' + fname + '_diff_black_only.png')
    plt.figure()
    pbar.update(1)  # ???????????????????????????
pbar.close()  # ???????????????????????????

ave_diff_white_only = ave_diff_white_only / 18
ave_diff_black_only = ave_diff_black_only / 18

for col_i, col in enumerate(color):
    plt.plot(ave_diff_white_only[:, col_i], color=col)
    ave_histr[:, col_i] += ave_white_only[:, col_i]
plt.xlim([-2, 256 + 2])
plt.ylim([0, 30000000])
plt.title(white_folder + '/' + 'ave_diff_white_only')
plt.savefig(white_folder + '/' + 'ave_white_only.png')
plt.figure()

for col_i, col in enumerate(color):
    plt.plot(ave_diff_black_only[:, col_i], color=col)
    ave_histr_inv[:, col_i] += ave_black_only[:, col_i]
plt.xlim([-2, 256 + 2])
plt.ylim([0, 7500000])
plt.title(black_folder + '/' + 'ave_diff_black_only')
plt.savefig(black_folder + '/' + 'ave_black_only.png')
plt.figure()
pbar.update(1)  # ???????????????????????????