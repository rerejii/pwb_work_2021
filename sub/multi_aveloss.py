import os
import glob
from tqdm import tqdm
import csv
import subprocess
import shutil
import sys
import math
import numpy as np
import pandas as pd
from natsort import natsorted
import matplotlib.pyplot as plt

# nas_path = '/home/hayakawa/share/nas/'
nas_path = 'Z:/hayakawa/'
img_name = 'weight_loss'
out_dir = 'binary/20201218/plot'
directory_set = [
    'binary/20201218/unet_fin-none_logit-true_thr-d0',
    'binary/20201218/unet_weight_fin-none_logit-true_thr-d0',
    'binary/20201218/unet_weight-d5-1_fin-none_logit-true_thr-d0',
    'binary/20201218/unet_weight-d10-1_fin-none_logit-true_thr-d0',
    'binary/20201218/unet_weight-d100-1_fin-none_logit-true_thr-d0',
    'binary/20201218/unet_weight-1-5_fin-none_logit-true_thr-d0',
    'binary/20201218/unet_weight-1-10_fin-none_logit-true_thr-d0',
    # 'binary/20201218/unet_fin-none_logit-true_thr-d0',
    # 'binary/20201218/unet_l2e-4_bd5_fin-none_logit-true_thr-d0',
]
label_set = [
    'no-weight',
    '[0.5:1]',
    '[0.01:1]',
    '[0.001:1]',
    '[1:5]',
    '[1:10]',
    # '0001',
    # 'l2e-4_bd5',
]
loc_set = ['upper right', 'upper right', 'upper right',]
epoch = 80
lim_min = 1.0
lim_max = 0.0

directory_set = [nas_path + directory_set[i] for i in range(len(directory_set))]

train_ave_list_set = []
test_ave_list_set = []
valid_ave_list_set = []
train_max_set = []
test_max_set = []
valid_max_set = []

for dir_i in range(len(directory_set)):
    directory = directory_set[dir_i]
    csv_files = glob.glob(directory + '/*/CsvDatas/step_accuracy.csv')
    if not csv_files:
        print('no file')
        sys.exit()
    csv_files = natsorted(csv_files)
    train_ave_list = []
    test_ave_list = []
    valid_ave_list = []
    train_max_list = []
    test_max_list = []
    valid_max_list = []
    os.makedirs(nas_path + out_dir, exist_ok=True)
    out_csvpath = nas_path + out_dir + '/' + os.path.basename(directory) + '.csv'
    with open(out_csvpath, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['ds', 'max_test_loss_index', 'train_loss', 'test_loss', 'valid_loss'])
        for i in range(len(csv_files)):
            ds_name = os.path.dirname(os.path.dirname(csv_files[i]))[-1]
            df = pd.read_csv(csv_files[i])

            # test max val
            max_test_acc = np.max(df['test_gen_loss'])
            max_test_acc_i = df['test_gen_loss'].idxmax()
            ado_train_acc = df['train_gen_loss'][max_test_acc_i]
            ado_valid_acc = df['valid_gen_loss'][max_test_acc_i]
            writer.writerow([ds_name, max_test_acc_i, ado_train_acc, max_test_acc, ado_valid_acc])

            # グラフ線用
            train_ave_list.append(df['train_gen_loss'].values[:epoch])
            test_ave_list.append(df['test_gen_loss'].values[:epoch])
            valid_ave_list.append(df['valid_gen_loss'].values[:epoch])

            # max
            train_max_list.append(ado_train_acc)
            test_max_list.append(max_test_acc)
            valid_max_list.append(ado_valid_acc)

        total_train_ave_list = np.mean(np.array(train_ave_list), axis=0)
        total_test_ave_list = np.mean(np.array(test_ave_list), axis=0)
        total_valid_ave_list = np.mean(np.array(valid_ave_list), axis=0)

        total_train_max = np.mean(np.array(train_max_list))
        total_test_max = np.mean(np.array(test_max_list))
        total_valid_max = np.mean(np.array(valid_max_list))

        writer.writerow(['max_ave', '-', total_train_max, total_test_max, total_valid_max])

        # 
        train_ave_list_set.append(total_train_ave_list)
        test_ave_list_set.append(total_test_ave_list)
        valid_ave_list_set.append(total_valid_ave_list)
        train_max_set.append(total_train_max)
        test_max_set.append(total_test_max)
        valid_max_set.append(total_valid_max)

ave_list_set = [train_ave_list_set, test_ave_list_set, valid_ave_list_set]
max_set = [train_max_set, test_max_set, valid_max_set]
task_set = ['train', 'test', 'validation',]
for task_i in range(len(task_set)):
    task = task_set[task_i]
    loc = loc_set[task_i]
    title = task+' loss (6ds ave)'
    item = task+'_loss'
    outpath = nas_path + out_dir + '/' + img_name + '_' + task + '.png'

    for dir_i in range(len(directory_set)):
        m = round( max_set[task_i][dir_i] , 5)
        label = label_set[dir_i] + '(' + str(m) + ')'
        x = list(range(1, epoch+1))
        plt.plot(x, ave_list_set[task_i][dir_i], label=label)
    plt.xlabel("Evaluation step")
    plt.ylabel("Loss")
    plt.title(title)
    plt.ylim([0., 0.008])
    plt.xlim([0,epoch+1])
    plt.legend(title="name(ds ave(max acc))", loc=loc)
    plt.savefig(outpath)
    plt.show()

        

#     for s in range(len(directory_set)):








        # print(max_test_acc_i)
        # print(max_test_acc)
        # print(df['test_accuracy'][max_test_acc_i])
        # ave_list.append(df[item].values[:epoch])
        # max_list.append(np.max(df[item].values) )


