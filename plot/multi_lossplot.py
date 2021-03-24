import os
import glob
import csv
import sys
import numpy as np
import pandas as pd
from natsort import natsorted
import matplotlib.pyplot as plt
import matplotlib

directory_set = [
    'binary/20210227/unet_use-bias_beta',
    'binary/20210227/unet_use-bias_beta_otsu',
]
guide_set = [
    'U-Net(RGB)',
    'U-Net(RGB+Otsu)',
]
csv_name = [
    'train_loss.csv',
    'train_loss.csv',
    'train_loss.csv',
    'train_loss.csv',
]
loc_set = ['lower right', 'lower right', 'lower right','lower right', 'lower right', 'lower right',]
plot_label = ['loss',]


epoch = 140
lim_min = 1.0
lim_max = 0.0
nas_path = 'Z:/hayakawa/'
out_dir = 'binary/20210227/plot'
img_name = 'loss2'
directory_set = [nas_path + directory_set[i] for i in range(len(directory_set))]

total_ave_list_set = np.zeros([len(plot_label), len(directory_set), epoch])  # ラベル単位, 条件単位, ステップ
total_max_list_set = np.zeros([len(plot_label), len(directory_set)])

for dir_i in range(len(directory_set)):  # 条件単位
    directory = directory_set[dir_i]
    csv_files = glob.glob(directory + '/*/CsvDatas/' + csv_name[dir_i])
    if not csv_files:
        print(directory + '/*/CsvDatas/' + csv_name[dir_i])
        print('no file')
        sys.exit()
    csv_files = natsorted(csv_files)
    os.makedirs(nas_path + out_dir, exist_ok=True)
    out_csvpath = nas_path + out_dir + '/' + os.path.basename(directory) + '.csv'
    ave_list_set = np.zeros([len(plot_label), len(csv_files), epoch])  # ラベル数, ds数, ステップ数
    max_list_set = np.zeros([len(plot_label), len(csv_files)])
    for i in range(len(csv_files)):  # ds単位
        df = pd.read_csv(csv_files[i])
        print(csv_files[i])
        # record = [ds_name, max_test_acc_i]
        for label_index in range(len(plot_label)):  # ラベル単位
            ave_list_set[label_index, i, :] = df[plot_label[label_index]].values[:epoch]
    for label_index in range(len(plot_label)):  # ラベル単位
        total_ave_list_set[label_index, dir_i, :] = np.mean(ave_list_set[label_index, :, :], axis=0)  # ds毎の平均取得

for label_index in range(len(plot_label)):
    label = plot_label[label_index]
    loc = loc_set[label_index]
    title = label + ' (6ds ave)'
    item = label + '_accuracy'
    outpath = nas_path + out_dir + '/' + img_name + '_' + label + '.png'
    for dir_i in range(len(directory_set)):
        # m = round(total_max_list_set[label_index][dir_i], 5)
        guide = guide_set[dir_i]
        # guide = guide_set[dir_i]
        x = list(range(1, epoch + 1))
        plt.plot(x, total_ave_list_set[label_index, dir_i, :], label=guide)
    plt.xlabel("Evaluation step")
    plt.ylabel("Loss")
    plt.title(title)
    plt.ylim([0., 0.05])
    plt.xlim([0, epoch + 1])
    # plt.legend(title="name(ds ave(max test index acc))", loc=loc)
    plt.legend(title="name", loc=loc)
    plt.savefig(outpath)
    plt.show()






