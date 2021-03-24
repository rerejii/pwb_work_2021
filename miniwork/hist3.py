import cv2
import numpy as np
import matplotlib
import os
from matplotlib import pyplot as plt
from tqdm import tqdm
import pandas as pd
import glob
from natsort import natsorted
matplotlib.use('Agg')

color = ('b','g','r')
one_csv_path_list = glob.glob('?????.csv')
ave_histr = np.zeros([256, 3])
pbar = tqdm(total=len(one_csv_path_list))
folder = 'normal'
os.makedirs(folder, exist_ok=True)
# ymax = 0
# for path_i, path in enumerate(one_csv_path_list):
#     one_histr = pd.read_csv(path, index_col=0)
#     m = np.amax(np.array(one_histr))
#     print(m)
#     if ymax < m:
#         ymax = m
# print(ymax)
# # sys.exit()


for path_i, path in enumerate(one_csv_path_list):
    fname = os.path.splitext(os.path.basename(path))[0]
    one_histr = np.zeros([256, 3])
    for col_i, col in enumerate(color):
        # img = cv2.imread(path)
        # one_histr[:, col_i] = cv2.calcHist([img],[col_i],None,[256],[0,256])[:, 0]
        one_histr = pd.read_csv(path, index_col=0).values
        ave_histr[:, col_i] += one_histr[:, col_i]
        plt.plot(one_histr[:, col_i], color=col)
    plt.xlim([-2, 256+2])
    plt.ylim([-10, 30000000])
    plt.title(fname)
    plt.savefig(folder + '/' + fname + '.png')
    plt.figure()
    # df = pd.DataFrame(one_histr, columns=color)
    # df.to_csv(fname + '.csv', sep=',')
    # np.savetxt(fname + '.csv', one_histr, delimiter=",")
    pbar.update(1)  # プロセスバーを進行
pbar.close()  # プロセスバーの終了
ave_histr = ave_histr / 18

for col_i, col in enumerate(color):
    plt.plot(ave_histr[:, col_i], color=col)
    plt.xlim([-2, 256+2])
    plt.ylim([-10, 30000000])
plt.xlim([0, 256])
# plt.ylim([-10, 330000000])
plt.title('ave_histr')
plt.savefig(folder + '/' + 'ave_histr.png')
plt.figure()
# df = pd.DataFrame(histr, columns=color)
# df.to_csv('ave_histr.csv', sep=',')
# np.savetxt('ave_histr.csv', histr, delimiter=",")



# for i,col in enumerate(color):
#     histr = cv2.calcHist([img],[i],None,[256],[0,256])
#     print(histr.shape)
#     plt.plot(histr,color = col)
#     plt.xlim([0,256])
# plt.show()