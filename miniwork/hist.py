import cv2
import numpy as np
import matplotlib
import os
from matplotlib import pyplot as plt
from tqdm import tqdm
from natsort import natsorted
import pandas as pd
matplotlib.use('Agg')

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
# img_path = [
#     'Sample_X011_Y064.png',
#     'Sample_X011_Y064.png',
#     'Sample_X011_Y064.png',
# ]

color = ('b','g','r')

histr = np.zeros([256, 3])
pbar = tqdm(total=len(img_path))
for path_i, path in enumerate(img_path):
    fname = os.path.splitext(os.path.basename(path))[0]
    print(fname)
    one_histr = np.zeros([256, 3])
    img = cv2.imread(path)
    for col_i, col in enumerate(color):
        one_histr[:, col_i] = cv2.calcHist([img],[col_i],None,[256],[0,256])[:, 0]
        histr[:, col_i] += one_histr[:, col_i]
        plt.plot(one_histr[:, col_i], color=col)
    plt.xlim([0, 256])
    # plt.ylim([-10, 330000000])
    plt.title(fname)
    plt.savefig(fname + '.png')
    plt.figure()
    df = pd.DataFrame(one_histr, columns=color)
    df.to_csv(fname + '.csv', sep=',')
    # np.savetxt(fname + '.csv', one_histr, delimiter=",")
    print(df)
    pbar.update(1)  # プロセスバーを進行

pbar.close()  # プロセスバーの終了
histr = histr / len(img_path)

for col_i, col in enumerate(color):
    plt.plot(histr[:, col_i], color=col)
    plt.xlim([0, 256])
    # plt.ylim([0, 1000])
plt.xlim([0, 256])
# plt.ylim([-10, 330000000])
plt.title('ave_histr')
plt.savefig('ave_histr.png')
plt.figure()
df = pd.DataFrame(histr, columns=color)
df.to_csv('ave_histr.csv', sep=',')
# np.savetxt('ave_histr.csv', histr, delimiter=",")



# for i,col in enumerate(color):
#     histr = cv2.calcHist([img],[i],None,[256],[0,256])
#     print(histr.shape)
#     plt.plot(histr,color = col)
#     plt.xlim([0,256])
# plt.show()