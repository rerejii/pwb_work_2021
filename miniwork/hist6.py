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


color = ('b','g','r')
ave_histr = np.zeros([256,3])
ave_histr_inv = np.zeros([256, 3])
csv_black_path_list = glob.glob('?????_black_only.csv')
csv_white_path_list = glob.glob('?????_white_only.csv')

pbar = tqdm(total=len(csv_black_path_list))
# for path_i, path in enumerate(img_path):
for path_i in range(len(csv_black_path_list)):
    black_path = csv_black_path_list[path_i]
    white_path = csv_white_path_list[path_i]
    fname = os.path.splitext(os.path.basename(black_path))[0][:5]
    one_histr = pd.read_csv(white_path, index_col=0).values
    one_histr_inv = pd.read_csv(black_path, index_col=0).values

    for col_i, col in enumerate(color):
        plt.plot(one_histr[:, col_i], color=col)
        ave_histr[:, col_i] += one_histr[:, col_i]
    plt.xlim([0, 256])
    plt.title(fname + '_white_only')
    plt.savefig(fname + '_white_only.png')
    plt.figure()

    for col_i, col in enumerate(color):
        plt.plot(one_histr_inv[:, col_i], color=col)
        ave_histr_inv[:, col_i] += one_histr_inv[:, col_i]
    plt.xlim([0, 256])
    plt.title(fname + '_black_only')
    plt.savefig(fname + '_black_only.png')
    plt.figure()
    pbar.update(1)  # プロセスバーを進行
pbar.close()  # プロセスバーの終了

ave_histr = ave_histr / len(csv_black_path_list)
ave_histr_inv = ave_histr_inv / len(csv_black_path_list)

for col_i, col in enumerate(color):
    plt.plot(ave_histr[:, col_i], color=col)
plt.xlim([0, 256])
plt.ylim([0, 10000000])
plt.title('ave_white_only')
plt.savefig('ave_white_only.png')
plt.figure()
df = pd.DataFrame(ave_histr, columns=color)
df.to_csv('ave_white_only.csv', sep=',')

for col_i, col in enumerate(color):
    plt.plot(ave_histr_inv[:, col_i], color=col)
plt.xlim([0, 256])
plt.ylim([0, 10000000])
plt.title('ave_black_only')
plt.savefig('ave_black_only.png')
plt.figure()
df = pd.DataFrame(ave_histr_inv, columns=color)
df.to_csv('ave_black_only.csv', sep=',')






