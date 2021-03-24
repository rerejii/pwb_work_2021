import os
import glob
import csv
import sys
import numpy as np
import pandas as pd
from natsort import natsorted
import matplotlib.pyplot as plt
import matplotlib
# matplotlib.use('Agg')

directory = 'binary/20210303/unet_rgb_otsu_loop'
guide = 'RGB'
title = 'U-Net(RGB+Otsu)loop'
csv_name = 'eva_all.csv'
plot_label = ['train_accuracy', 'test_accuracy', 'validation_accuracy']
epoch = 120
nas_path = 'Z:/hayakawa/'
out_dir = 'binary/20210303/plot'
total_ave_list = np.zeros([len(plot_label), epoch])
csv_files = glob.glob(nas_path + directory + '/*/CsvDatas/' + csv_name)
if not csv_files:
    print(nas_path + '/*/CsvDatas/' + csv_name)
    print('no file')
    sys.exit()
csv_files = natsorted(csv_files)
os.makedirs(nas_path + out_dir, exist_ok=True)
ave_list = np.zeros([len(plot_label), len(csv_files), epoch])  # ラベル数, ds数, ステップ数
for ds_index in range(len(csv_files)):  # ds単位
    print(csv_files[ds_index])
    ds_name = os.path.dirname(os.path.dirname(csv_files[ds_index]))[-1]
    df = pd.read_csv(csv_files[ds_index])
    for label_index in range(len(plot_label)):  # ラベル単位
        ave_list[label_index, ds_index, :] = df[plot_label[label_index]].values[:epoch]
for label_index in range(len(plot_label)):  # ラベル単位
    total_ave_list[label_index, :] = np.mean(ave_list[label_index, :, :], axis=0)

outpath = nas_path + out_dir + '/' + title + '.png'
x = list(range(1, epoch + 1))
for label_index in range(len(plot_label)):  # ラベル単位
    plt.plot(x, total_ave_list[label_index, :], label=plot_label[label_index])
plt.xlabel("Evaluation step")
plt.ylabel("Accuracy")
plt.title(title + ' 6 dataset average')
plt.ylim([0.97, 1])
plt.xlim([0, epoch + 1])
loc = ['train', 'test', 'validation']
plt.legend(title="accuracy", loc='lower right')
# plt.legend(title="name", loc=loc)
plt.savefig(outpath)
plt.show()