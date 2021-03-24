import numpy as np
import tensorflow as tf
from tqdm import tqdm
import sys
import csv
import os
import cv2
from natsort import natsorted
import pandas as pd

img_path_root = 'Z:/hayakawa/share/'
image_paths = [
    'img/17H-0863-1_L0001/L0001.png',
    'img/17H-0863-1_L0002-old/L0002.png',
    'img/17H-0863-1_L0003/L0003.png',
    'img/17H-0863-1_L0004/L0004.png',
    'img/17H-0863-1_L0005/L0005.png',
    'img/17H-0863-1_L0006-new/L0006.png',
    'img/17H-0863-1_L0007/L0007.png',
    'img/17H-0863-1_L0008/L0008.png',
    'img/17H-0863-1_L0009/L0009.png',
    'img/17H-0863-1_L0010/L0010.png',
    'img/17H-0863-1_L0011/L0011.png',
    'img/17H-0863-1_L0012/L0012.png',
    'img/17H-0863-1_L0013/L0013.png',
    'img/17H-0863-1_L0014/L0014.png',
    'img/17H-0863-1_L0015/L0015.png',
    'img/17H-0863-1_L0016/L0016.png',
    'img/17H-0863-1_L0017/L0017.png',
    'img/17H-0863-1_L0018/L0018.png',
]

image_paths = [img_path_root + path for path in image_paths]
image_paths = natsorted(image_paths)

with open('one_std.csv', 'w') as f:
    writer = csv.writer(f, lineterminator='\n')
    writer.writerow(['filename', 'R_std', 'R_mean', 'G_std', 'G_mean', 'B_std', 'B_mean'])
    for path_i, path in enumerate(image_paths):
        fname = os.path.splitext(os.path.basename(path))[0]
        img = cv2.imread(path)
        R_data = img[:, :, 2].ravel()
        G_data = img[:, :, 1].ravel()
        B_data = img[:, :, 0].ravel()

        R_std = np.std(R_data)
        R_mean = np.mean(R_data)
        print(R_std)
        print(R_mean)

        G_std = np.std(G_data)
        G_mean = np.mean(G_data)
        print(G_std)
        print(G_mean)

        B_std = np.std(B_data)
        B_mean = np.mean(B_data)
        print(B_std)
        print(B_mean)

        writer.writerow([fname, R_std, R_mean, G_std, G_mean, B_std, B_mean])


