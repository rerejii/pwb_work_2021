import cv2
import numpy as np
import matplotlib
import os
from matplotlib import pyplot as plt
from tqdm import tqdm
from natsort import natsorted
import glob
import pandas as pd

matplotlib.use('Agg')
color = ('b','g','r')
img_path_root = 'Z:/hayakawa/share/'
img_path = [
    'img/17H-0863-1_L0006-new/L0006.png',
    'img/17H-0863-1_L0011/L0011.png',
    'img/17H-0863-1_L0017/L0017.png',
    'img/17H-0863-1_L0003/L0003.png',
    'img/17H-0863-1_L0005/L0005.png',
    'img/17H-0863-1_L0009/L0009.png',
    'img/17H-0863-1_L0002-old/L0002.png',
    'img/17H-0863-1_L0008/L0008.png',
    'img/17H-0863-1_L0013/L0013.png',
    'img/17H-0863-1_L0007/L0007.png',
    'img/17H-0863-1_L0012/L0012.png',
    'img/17H-0863-1_L0014/L0014.png',
    'img/17H-0863-1_L0004/L0004.png',
    'img/17H-0863-1_L0015/L0015.png',
    'img/17H-0863-1_L0016/L0016.png',
    'img/17H-0863-1_L0001/L0001.png',
    'img/17H-0863-1_L0010/L0010.png',
    'img/17H-0863-1_L0018/L0018.png',
]
img_path = [img_path_root + path for path in img_path]
img_path = natsorted(img_path)
one_csv_path_list = glob.glob('?????.csv')
ave_csv_path = 'ave_histr.csv'
ave_histr = pd.read_csv(ave_csv_path, index_col=0).values
for path_i, one_csv_path in enumerate(one_csv_path_list):
    fname = os.path.splitext(os.path.basename(one_csv_path))[0]
    histr = pd.read_csv(one_csv_path, index_col=0).values
    diff_histr = histr - ave_histr
    for col_i, col in enumerate(color):
        plt.plot(diff_histr[:, col_i], color=col)
    plt.xlim([0, 256])
    plt.ylim([-15000000, 15000000])
    plt.savefig(fname + '_diff.png')
    df = pd.DataFrame(diff_histr, columns=color)
    df.to_csv(fname + '_diff.csv', sep=',')
    plt.figure()


