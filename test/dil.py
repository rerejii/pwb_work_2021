import glob
import numpy as np
import os
import sys
from tqdm import tqdm
from natsort import natsorted
import cv2
import pandas as pd

# args = sys.argv
# DEVICE_NUMBER_STR = args[1]  # 使用するGPU設定
# DEVICE_STR_LIST = DEVICE_NUMBER_STR.split(',')
# DEVICE_LIST = [int(n) for n in range(len(DEVICE_STR_LIST))]
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = DEVICE_NUMBER_STR


import tensorflow as tf

# # 評価画像の出力
boundary_kernel = 5
crush_kernel = 9
black = [0, 0, 0]  # 予測:0黒 解答:0黒
white = [255, 255, 255]  # 予測:1白 解答:1白
red = [255, 0, 0]  # 予測:1白 解答:0黒
blue = [0, 204, 255]  # 予測:0黒 解答:1白

out_folder = 'crop_img/dil'
os.makedirs(out_folder, exist_ok=True)

img_path_root = 'Z:/hayakawa/share/'
img_path = [
    'img/17H-0863-1_L0006-new/L0006_bin.png',
    'img/17H-0863-1_L0011/L0011_bin.png',
    'img/17H-0863-1_L0017/L0017_bin.png',
    'img/17H-0863-1_L0003/L0003_bin.png',
    'img/17H-0863-1_L0005/L0005_bin.png',
    'img/17H-0863-1_L0009/L0009_bin.png',
    'img/17H-0863-1_L0002-old/L0002_bin.png',
    'img/17H-0863-1_L0008/L0008_bin.png',
    'img/17H-0863-1_L0013/L0013_bin.png',
    'img/17H-0863-1_L0007/L0007_bin.png',
    'img/17H-0863-1_L0012/L0012_bin.png',
    'img/17H-0863-1_L0014/L0014_bin.png',
    'img/17H-0863-1_L0004/L0004_bin.png',
    'img/17H-0863-1_L0015/L0015_bin.png',
    'img/17H-0863-1_L0016/L0016_bin.png',
    'img/17H-0863-1_L0001/L0001_bin.png',
    'img/17H-0863-1_L0010/L0010_bin.png',
    'img/17H-0863-1_L0018/L0018_bin.png',
]
img_path = [img_path_root + path for path in img_path]
img_path = natsorted(img_path)

pbar = tqdm(total=len(img_path))
for path_i, path in enumerate(img_path):
    img = cv2.imread(path, 0)
    fname = os.path.splitext(os.path.basename(path))[0]
    inv = cv2.bitwise_not(img)
    dst = cv2.distanceTransform(img, cv2.DIST_L2, 5)
    inv_dst = cv2.distanceTransform(inv, cv2.DIST_L2, 5)
    dst = dst + inv_dst
    dst = np.array(dst, np.int)
    dst[dst > 255] = 255
    print('output now')
    cv2.imwrite(out_folder + '/' + fname + '_dst.png', dst)
    # df = pd.DataFrame(dst)
    # print('output now')
    # df.to_csv(out_folder + '/' + fname + '_dst.csv', sep=',', header=False, index=False)
    pbar.update(1)  # プロセスバーを進行
pbar.close()  # プロセスバーの終了

