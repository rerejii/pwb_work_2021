import os
import glob
import csv
import sys
import numpy as np
import pandas as pd
from natsort import natsorted
import matplotlib.pyplot as plt

directory_set = [
    'binary/20201218/unet_fin-none_logit-true_thr-d0',
    'binary/20201218/unet_weight-d5-1_fin-none_logit-true_thr-d0',
    'binary/20201218/unet_weight-d01-1_fin-none_logit-true_thr-d0',
    'binary/20201218/unet_weight-d001-1_fin-none_logit-true_thr-d0',
]
guide_set = [
    'None',
    'd5-1',
    'd01-1',
    'd001-1',
]
csv_name = [
    'acc_ckpt.csv',
    'acc_ckpt.csv',
    'acc_ckpt.csv',
    'step_accuracy.csv',
]

loc_set = ['lower right', 'lower right', 'lower right',
           'lower right', 'lower right', 'lower right',
           'lower right', 'lower right', 'lower right',
           'lower right', 'lower right', 'lower right',]
plot_label = ['train_accuracy', 'test_accuracy', 'validation_accuracy',
              'train_weight_only_accracy', 'test_weight_only_accracy', 'valid_weight_only_accracy',
              'train_boundary_removal_accracy', 'test_boundary_removal_accracy', 'valid_boundary_removal_accracy',
              'train_boundary_accracy', 'test_boundary_accracy', 'valid_boundary_accracy',
              ]
epoch = 76
lim_min = 1.0
lim_max = 0.0
nas_path = 'Z:/hayakawa/'
out_dir = 'binary/20201218/plot'
img_name = 'weight'
directory_set = [nas_path + directory_set[i] for i in range(len(directory_set))]

# ave_list_set = [[] for i in range(plot_label)]
# max_list_set = [[] for i in range(plot_label)]

total_ave_list_set = np.zeros([len(plot_label), len(directory_set), epoch])  # ラベル単位, 条件単位, ステップ
total_max_list_set = np.zeros([len(plot_label), len(directory_set)])

for dir_i in range(len(directory_set)):  # 条件単位
    directory = directory_set[dir_i]
    csv_files = glob.glob(directory + '/*/CsvDatas/' + csv_name[dir_i])
    if not csv_files:
        print('no file')
        sys.exit()
    csv_files = natsorted(csv_files)
    os.makedirs(nas_path + out_dir, exist_ok=True)
    out_csvpath = nas_path + out_dir + '/' + os.path.basename(directory) + '.csv'
    ave_list_set = np.zeros([len(plot_label), len(csv_files), epoch])  # ラベル数, ds数, ステップ数
    max_list_set = np.zeros([len(plot_label), len(csv_files)])
    with open(out_csvpath, 'w') as f:
        writer = csv.writer(f)
        label = ['ds', 'max_test_accuracy_index']
        label[len(label):len(label)] = plot_label
        writer.writerow(label)
        for i in range(len(csv_files)):  # ds単位
            ds_name = os.path.dirname(os.path.dirname(csv_files[i]))[-1]
            df = pd.read_csv(csv_files[i])
            max_test_acc = np.max(df['test_accuracy'])
            max_test_acc_i = df['test_accuracy'].idxmax()
            record = [ds_name, max_test_acc_i]
            for label_index in range(len(plot_label)):  # ラベル単位
                ave_list_set[label_index, i, :] = df[plot_label[label_index]].values[:epoch]
                max_list_set[label_index, i] = df[plot_label[label_index]][max_test_acc_i]
                record.append(df[plot_label[label_index]][max_test_acc_i])
            writer.writerow(record)
        for label_index in range(len(plot_label)):  # ラベル単位
            total_ave_list_set[label_index, dir_i, :] = np.mean(ave_list_set[label_index, :, :], axis=0)  # ds毎の平均取得
            total_max_list_set[label_index, dir_i] = np.mean(max_list_set[label_index])

for label_index in range(len(plot_label)):
    label = plot_label[label_index]
    loc = loc_set[label_index]
    title = label + ' accuracy (6ds ave)'
    item = label + '_accuracy'
    outpath = nas_path + out_dir + '/' + img_name + '_' + label + '.png'
    for dir_i in range(len(directory_set)):
        m = round(total_max_list_set[label_index][dir_i], 5)
        guide = guide_set[dir_i] + '(' + str(m) + ')'
        x = list(range(1, epoch + 1))
        plt.plot(x, total_ave_list_set[label_index, dir_i, :], label=guide)
    plt.xlabel("Evaluation step")
    plt.ylabel("Accuracy")
    plt.title(title)
    plt.ylim([0.7, 1])
    plt.xlim([0, epoch + 1])
    plt.legend(title="name(ds ave(max test index acc))", loc=loc)
    plt.savefig(outpath)
    plt.show()




