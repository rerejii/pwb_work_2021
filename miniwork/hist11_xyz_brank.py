import cv2
import numpy as np
import matplotlib
import os
from matplotlib import pyplot as plt
from tqdm import tqdm
from natsort import natsorted
import pandas as pd
matplotlib.use('Agg')
brank = [546+100, 347+100, 330+100, 319+100]
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
ans_path = [
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
ans_path = [img_path_root + path for path in ans_path]
ans_path = natsorted(ans_path)

color = ('b','g','r')
xyz = ('x','y','z')
ave_histr = np.zeros([256,3])
ave_histr_inv = np.zeros([256, 3])
pbar = tqdm(total=len(img_path))
for path_i, path in enumerate(img_path):
    fname = os.path.splitext(os.path.basename(path))[0]
    one_histr = np.zeros([256, 3])
    one_histr_inv = np.zeros([256, 3])
    img = cv2.imread(path)
    img = img[brank[0]:-1 * brank[1], brank[2]:-1 * brank[3], :]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2XYZ)
    mask = cv2.imread(ans_path[path_i], 0)
    mask = mask[brank[0]:-1 * brank[1], brank[2]:-1 * brank[3]]
    mask_inv = cv2.bitwise_not(mask)
    cut_img = np.zeros_like(img)
    cut_img_inv = np.zeros_like(img)
    for i in range(3):
        cut_img[:, :, i] = img[:, :, i]
        cut_img_inv[:, :, i] = img[:, :, i]

    for col_i, col in enumerate(color):
        one_histr[:, col_i] = cv2.calcHist([cut_img], [col_i], mask, [256], [0, 256])[:, 0]
        ave_histr[:, col_i] += one_histr[:, col_i]
        plt.plot(one_histr[:, col_i], color=col)
    plt.xlim([0, 256])
    plt.title(fname + '_white_only_xyz_brank')
    plt.savefig(fname + '_white_only_xyz_brank.png')
    plt.figure()
    df = pd.DataFrame(one_histr, columns=xyz)
    df.to_csv(fname + '_white_only_xyz_brank.csv', sep=',')

    for col_i, col in enumerate(color):
        one_histr_inv[:, col_i] = cv2.calcHist([cut_img_inv], [col_i], mask_inv, [256], [0, 256])[:, 0]
        ave_histr_inv[:, col_i] += one_histr_inv[:, col_i]
        plt.plot(one_histr_inv[:, col_i], color=col)
    plt.xlim([0, 256])
    plt.title(fname + '_black_only_xyz_brank')
    plt.savefig(fname + '_black_only_xyz_brank.png')
    plt.figure()
    df = pd.DataFrame(one_histr_inv, columns=xyz)
    df.to_csv(fname + '_black_only_xyz_brank.csv', sep=',')

    pbar.update(1)  # プロセスバーを進行
pbar.close()  # プロセスバーの終了
ave_histr = ave_histr / len(img_path)
ave_histr_inv = ave_histr_inv / len(img_path)

for col_i, col in enumerate(color):
    plt.plot(ave_histr[:, col_i], color=col)
plt.xlim([0, 256])
plt.title('ave_white_only_xyz_brank')
plt.savefig('ave_white_only_xyz_brank.png')
plt.figure()
df = pd.DataFrame(ave_histr, columns=xyz)
df.to_csv('ave_white_only_xyz_brank.csv', sep=',')

for col_i, col in enumerate(color):
    plt.plot(ave_histr_inv[:, col_i], color=col)
plt.xlim([0, 256])
plt.title('ave_black_only_xyz_brank')
plt.savefig('ave_black_only_xyz_brank.png')
plt.figure()
df = pd.DataFrame(ave_histr_inv, columns=xyz)
df.to_csv('ave_black_only_xyz_brank.csv', sep=',')






