import cv2
import numpy as np
import matplotlib
import os
from matplotlib import pyplot as plt
from tqdm import tqdm
from natsort import natsorted
import pandas as pd
import glob
import csv

sample_img = [
    'generator_L0001.png',
    'generator_L0002.png',
    'generator_L0003.png',
    'generator_L0004.png',
    'generator_L0005.png',
    'generator_L0006.png',
    'generator_L0007.png',
    'generator_L0008.png',
    'generator_L0009.png',
    'generator_L0010.png',
    'generator_L0011.png',
    'generator_L0012.png',
    'generator_L0013.png',
    'generator_L0014.png',
    'generator_L0015.png',
    'generator_L0016.png',
    'generator_L0017.png',
    'generator_L0018.png',
]

bin_img = [
    'L0001_bin.png',
    'L0002_bin.png',
    'L0003_bin.png',
    'L0004_bin.png',
    'L0005_bin.png',
    'L0006_bin.png',
    'L0007_bin.png',
    'L0008_bin.png',
    'L0009_bin.png',
    'L0010_bin.png',
    'L0011_bin.png',
    'L0012_bin.png',
    'L0013_bin.png',
    'L0014_bin.png',
    'L0015_bin.png',
    'L0016_bin.png',
    'L0017_bin.png',
    'L0018_bin.png',
]

ROOT_PATH = 'Z:/hayakawa/binary/20210128/'
TARGET = 'unet_use-bias_beta_otsu2_deep'
BIN_PATH = 'Z:/hayakawa/work20/dataset2/img'

sam_path_set = natsorted(glob.glob(ROOT_PATH + TARGET + '/*/generator_L00??.png'))
ans_path_set = natsorted(glob.glob(BIN_PATH + '/*/L00??_bin.png'))

with open(ROOT_PATH + TARGET + '/' + TARGET + '_acc.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(['name', 'acc'])
    total_acc = 0.
    for i in range(len(sam_path_set)):
        sam_path = glob.glob(ROOT_PATH + TARGET + '/*/' + sample_img[i])[0]
        ans_path = glob.glob(BIN_PATH + '/*/' + bin_img[i])[0]
        print('Image %d / %d' % (i + 1, len(sam_path_set)))
        print(sam_path)
        print(ans_path)
        in_img = cv2.imread(sam_path, 0)
        ans_img = cv2.imread(ans_path, 0)
        h, w = in_img.shape
        acc = np.sum(in_img == ans_img) / (h * w)
        print(acc)
        writer.writerow([os.path.basename(sam_path), acc])
        total_acc += acc
    total_acc = total_acc / len(sam_path_set)
    print('total acc: ' + str(total_acc))
    writer.writerow(['total ave acc', total_acc])