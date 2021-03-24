import os
import glob
from tqdm import tqdm
import csv
import subprocess
import cv2
import shutil
import sys
import math
import numpy as np
import pandas as pd
from natsort import natsorted
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import matplotlib.ticker as ticker

nas_path = 'Z:/hayakawa/'
img_name = 'dsc-A'
directory = 'binary/20201218/unet_fin-none_logit-true_thr-d0'
name = 'unet_fin-none_logit-true_thr-d0-A'
total_ckpt = 80
ckpt_step = 21888
lim_min = 1.0
lim_max = 0.0

directory = nas_path + directory

train_ave_list_set = []
test_ave_list_set = []
valid_ave_list_set = []
dsc_list = []
df_list = []
# train_dsc_list_set = []
# test_dsc_list_set = []
# valid_dsc_list_set = []
# train_max_set = []
# test_max_set = []
# valid_max_set = []

csv_files = glob.glob(directory + '/'+name+'/CsvDatas/step_accuracy.csv')
# dsc_csv = glob.glob(directory + '/*/CsvDatas/eva_ckpt_L????.csv')
dsc_csv = []
for index in range(18):
    i = index + 1
    if i < 10:
        s = '000'+str(i)
    else:
        s = '00'+str(i)
    if not os.path.exists(directory + '/'+name+'/CsvDatas/eva_ckpt_L'+s+'.csv'):
        continue
    dsc_csv.append(glob.glob(directory + '/'+name+'/CsvDatas/eva_ckpt_L'+s+'.csv')[0])

if not csv_files:
    print('no file')
    sys.exit()

csv_files = natsorted(csv_files)
# dsc_csv = natsorted(dsc_csv)
os.makedirs( nas_path + 'binary/20201218/plot', exist_ok=True)
out_csvpath = nas_path + 'binary/20201218/plot/'+os.path.basename(directory)+'_dsc.csv'

min_index = total_ckpt
for dsc_i in range(len(dsc_csv)):
    df_list.append(pd.read_csv(dsc_csv[dsc_i]))
    df = df_list[dsc_i]
    if df.shape[0] < min_index:
        min_index = df.shape[0]

for step_i in range(total_ckpt):
    if step_i >= min_index:
        dsc_list.append(None)
        continue

    li = []
    for dsc_i in range(len(dsc_csv)):
        df = df_list[dsc_i]
        if df['total'][step_i] is None:
            li = None
            break
        else:
            li.append(df['total'][step_i])
    if li is not None:
        dsc_list.append(np.mean(np.array(li), axis=0))
    else:
        dsc_list.append(None)
dsc_list.reverse()
print(dsc_list)

train_ave_list = []
test_ave_list = []
valid_ave_list = []
train_max_list = []
test_max_list = []
valid_max_list = []
for i in range(len(csv_files)):
    ds_name = os.path.dirname(os.path.dirname(csv_files[i]))[-1]
    df = pd.read_csv(csv_files[i])

    # test max val
    max_test_acc = np.max(df['test_accuracy'])
    max_test_acc_i = df['test_accuracy'].idxmax()
    ado_valid_acc = df['validation_accuracy'][max_test_acc_i]
    # writer.writerow([ds_name, max_test_acc_i, ado_train_acc, max_test_acc, ado_valid_acc])

    # グラフ線用
    valid_ave_list.append(df['validation_accuracy'].values[:total_ckpt])

    # max
    valid_max_list.append(ado_valid_acc)

total_valid_ave_list = np.mean(np.array(valid_ave_list), axis=0)
total_valid_max = np.mean(np.array(valid_max_list))

task = 'validation'
loc = 'lower right'
title = ''
item = ''
outpath = nas_path + 'binary/20201218/plot/' + img_name + '.png'

# plot
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()

color_1 = "c"
color_2 = "r"
x = range(1, 21)
x = x[total_ckpt - min_index:]
total_valid_ave_list = total_valid_ave_list[total_ckpt - min_index:]
dsc_list = dsc_list[total_ckpt - min_index:]
# min_index



ax1.bar(x, dsc_list, color=color_1,
         label="dsc")
ax2.plot(x, total_valid_ave_list, color=color_2,
        label="accuracy")

# 軸の目盛りの最大値をしている
# axesオブジェクトに属するYaxisオブジェクトの値を変更
# ax1.yaxis.set_major_locator(MaxNLocator(nbins=5))
# ax2.yaxis.set_major_locator(MaxNLocator(nbins=5))

# 軸の縦線の色を変更している
# axesオブジェクトに属するSpineオブジェクトの値を変更
# 図を重ねてる関係で、ax2のみいじる。
ax2.spines['left'].set_color(color_1)
ax2.spines['right'].set_color(color_2)

# 軸の縦線の色を変更している
ax1.tick_params(axis='y', colors=color_1)
ax2.tick_params(axis='y', colors=color_2)

# 　軸の目盛りの単位を変更する
# ax1.yaxis.set_major_formatter(ticker.FormatStrFormatter("%d人"))
# ax2.yaxis.set_major_formatter(ticker.FormatStrFormatter("%d件"))

# 凡例
# グラフの本体設定時に、ラベルを手動で設定する必要があるのは、barplotのみ。plotは自動で設定される＞
handler1, label1 = ax1.get_legend_handles_labels()
handler2, label2 = ax2.get_legend_handles_labels()
# 凡例をまとめて出力する
ax1.legend(handler1 + handler2, label1 + label2, loc=loc, borderaxespad=0.)
plt.savefig(outpath)
plt.show()