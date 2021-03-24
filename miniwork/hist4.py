import cv2
import numpy as np
import matplotlib
import os
from matplotlib import pyplot as plt
from tqdm import tqdm
import pandas as pd
import glob
matplotlib.use('Agg')

color = ('b','g','r')
one_csv_path_list = glob.glob('?????.csv')
ave_histr = np.zeros([256, 3])
pbar = tqdm(total=len(one_csv_path_list))

ymax = 0
for path_i, path in enumerate(one_csv_path_list):
    one_histr = pd.read_csv(path, index_col=0)
    m = np.amax(np.array(one_histr))
    if ymax < m:
        ymax = m

fig = plt.figure()
ax_list = [fig.add_subplot(5, 4, i+1) for i in range(5*4)]

for path_i, path in enumerate(one_csv_path_list):
    fname = os.path.splitext(os.path.basename(path))[0]
    one_histr = np.zeros([256, 3])
    ax = ax_list[path_i]
    for col_i, col in enumerate(color):
        one_histr = pd.read_csv(path, index_col=0).values
        ave_histr[:, col_i] += one_histr[:, col_i]
        ax.plot(one_histr[:, col_i], color=col)
        ax.set_title(fname)
    pbar.update(1)  # プロセスバーを進行
pbar.close()  # プロセスバーの終了

ave_histr = ave_histr / len(one_csv_path_list)
ax = ax_list[-1]
for col_i, col in enumerate(color):
    ax.plot(ave_histr[:, col_i], color=col)
ax.set_title('ave_histr')
# plt.xlim([-2, 256 + 2])
# plt.ylim([int(-1*ymax*0.01), ymax + int(ymax*0.01)])
plt.savefig('all_hist.png')
plt.figure()