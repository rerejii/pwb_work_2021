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

directory_set = [
    # 'binary/20210128/unet_use-bias_beta',
    # 'binary/20210128/unet_use-bias_beta_2',
    # 'binary/20210227/unet_use-bias_beta',
    # 'binary/20210128/unet_use-bias_beta_otsu2',
    # 'binary/20210128/unet_use-bias_beta_otsu2_fixmask',
    # 'binary/20210227/unet_use-bias_beta',
    'binary/20210303/unet_rgb',
    # 'binary/20210303/unet_rgb_ls',
    'binary/20210303/unet_rgb_otsu',
    'binary/20210303/unet_rgb_otsu_loop',
    # 'binary/20210222/unet_use-bias_beta_otsu2_deep_multi-loss',
    # 'binary/20210222/unet_use-bias_beta_otsu_512'
]
guide_set = [
    'RGB',
    # 'RGB+LS',
    'RGB+Otsu',
    'RGB+Otsu Loop'
    # 'fix_mask',
    # 'deep',
    # '512',
]
csv_name = [
    # 'step_accuracy.csv',
    'eva_all.csv',
    'eva_all.csv',
    'eva_all.csv',
    'eva_all.csv',
]
# test_F1	test_Fd5	test_F2	validation_F1	validation_Fd5	validation_F2
loc_set = ['lower right', 'lower right', 'lower right', 'lower right', 'lower right', 'lower right',
           'lower right', 'lower right', 'lower right', 'lower right', 'lower right', 'lower right',
           'lower right', 'lower right', 'lower right', 'lower right', 'lower right', 'lower right',
           'lower right', 'lower right','lower right', 'lower right','lower right', 'lower right',]
plot_label = ['test_accuracy', 'validation_accuracy', 'train_accuracy'
              # 'test_F1', 'test_Fd5', 'test_F2',
              # 'validation_F1', 'validation_Fd5', 'validation_F2',
              # 'test_N_F1', 'test_N_Fd5', 'test_N_F2',
              # 'validation_N_F1', 'validation_N_Fd5', 'validation_N_F2',
              # 'test_IoU', 'validation_IoU', 'test_N_IoU', 'validation_N_IoU',
              # 'test_mean_IoU', 'validation_mean_IoU'
              ]
epoch = 120
lim_min = 1.0
lim_max = 0.0
nas_path = 'Z:/hayakawa/'
out_dir = 'binary/20210303/plot'
img_name = 'main'
directory_set = [nas_path + directory_set[i] for i in range(len(directory_set))]
# print(directory_set)
# ave_list_set = [[] for i in range(plot_label)]
# max_list_set = [[] for i in range(plot_label)]

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
    with open(out_csvpath, 'w') as f:
        writer = csv.writer(f)
        label = ['ds', 'max_test_accuracy_index']
        label[len(label):len(label)] = plot_label
        writer.writerow(label)
        for i in range(len(csv_files)):  # ds単位
            print(csv_files[i])
            ds_name = os.path.dirname(os.path.dirname(csv_files[i]))[-1]
            df = pd.read_csv(csv_files[i])
            max_test_acc = np.max(df['test_accuracy'][:epoch])
            max_test_acc_i = df['test_accuracy'][:epoch].idxmax()
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
    title = label + ' (6ds ave)'
    item = label + '_accuracy'
    outpath = nas_path + out_dir + '/' + img_name + '_' + label + '.png'
    for dir_i in range(len(directory_set)):
        m = round(total_max_list_set[label_index][dir_i], 5)
        guide = guide_set[dir_i] + '(' + str(m) + ')'
        # guide = guide_set[dir_i]
        x = list(range(1, epoch + 1))
        plt.plot(x, total_ave_list_set[label_index, dir_i, :], label=guide)
    plt.xlabel("Evaluation step")
    plt.ylabel("Accuracy")
    plt.title(title)
    plt.ylim([0.95, 1])
    plt.xlim([0, epoch + 1])
    # plt.legend(title="name(ds ave(max test index acc))", loc=loc)
    plt.legend(title="name", loc=loc)
    plt.savefig(outpath)
    plt.show()




