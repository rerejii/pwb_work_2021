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
tag = 'E'
img_name = 'dsc3-' + tag
directory = 'binary/20201218/unet_fin-none_logit-true_thr-d0'
name = 'unet_fin-none_logit-true_thr-d0'
acc_tag = 'valid_weight_only_accracy'
total_ckpt = 80
cut_step = 0
ckpt_step = 2188
lim_min = 1.0
lim_max = 0.0

directory = nas_path + directory

csv_files = glob.glob(directory + '/'+name+'-'+tag+'/CsvDatas/acc_ckpt.csv')
# dsc_csv = glob.glob(directory + '/*/CsvDatas/eva_ckpt_L????.csv')
dsc_csv = []
for index in range(18):
    i = index + 1
    if i < 10:
        s = '000'+str(i)
    else:
        s = '00'+str(i)
    if not os.path.exists(directory + '/'+name+'-'+tag+'/CsvDatas/eva_ckpt_L'+s+'.csv'):
        continue
    dsc_csv.append(glob.glob(directory + '/'+name+'-'+tag+'/CsvDatas/eva_ckpt_L'+s+'.csv')[0])

if not csv_files:
    print('no csv file')
    sys.exit()
if not dsc_csv:
    print('no dsc file')
    sys.exit()

csv_files = natsorted(csv_files)
# dsc_csv = natsorted(dsc_csv)
os.makedirs( nas_path + 'binary/20201218/plot', exist_ok=True)
out_csvpath = nas_path + 'binary/20201218/plot/'+os.path.basename(directory)+'_dsc.csv'

df_list = []
dsc_list = []

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
# print(dsc_list)


df = pd.read_csv(csv_files[0])
dsc_list = [9999 if np.isnan(dsc_list[i]) else dsc_list[i] for i in range(len(dsc_list))]
li = [[dsc_list[i], df[acc_tag][i], i] for i in range(len(dsc_list)) ]
# print(li)
li = sorted(li)
max_valid_acc_i = df[acc_tag].idxmax()
print(tag)
print(max_valid_acc_i)
print(df[acc_tag][max_valid_acc_i])

df = pd.DataFrame(li,
                  columns=['err', 'accuracy', 'sort_index'])
print(df)

df.to_csv(img_name+'.csv')


#
# train_ave_list = []
# test_ave_list = []
# valid_ave_list = []
# train_max_list = []
# test_max_list = []
# valid_max_list = []
# for i in range(len(csv_files)):
#     ds_name = os.path.dirname(os.path.dirname(csv_files[i]))[-1]
#     df = pd.read_csv(csv_files[i])
#
#     # test max val
#     max_test_acc = np.max(df['test_accuracy'])
#     max_test_acc_i = df['test_accuracy'].idxmax()
#     ado_valid_acc = df['validation_accuracy'][max_test_acc_i]
#     # writer.writerow([ds_name, max_test_acc_i, ado_train_acc, max_test_acc, ado_valid_acc])
#
#     # ???????????????
#     valid_ave_list.append(df['validation_accuracy'].values[:total_ckpt])
#
#     # max
#     valid_max_list.append(ado_valid_acc)
#
# total_valid_ave_list = np.mean(np.array(valid_ave_list), axis=0)
# total_valid_max = np.mean(np.array(valid_max_list))
#
# task = 'validation'
# loc = 'lower right'
# title = str(max_test_acc_i)
# item = ''
# outpath = nas_path + 'binary/20201218/plot/' + img_name + '.png'
#
# # plot
# fig, ax1 = plt.subplots()
# ax2 = ax1.twinx()
#
# color_1 = "c"
# color_2 = "m"
# x = range(total_ckpt+1)
# x = x[cut_step+1:]
# total_valid_ave_list = total_valid_ave_list[cut_step:]
# dsc_list = dsc_list[cut_step:]
# # min_index
#
#
#
# # ax1.plot(x, dsc_list, color=color_1,
# #          label="dsc")
# # ax2.plot(x, total_valid_ave_list, color=color_2,
# #         label="accuracy")
#
# ax1.bar(x, dsc_list, color=color_1,
#          label="dsc")
# ax2.plot(x, total_valid_ave_list, color=color_2,
#         label="accuracy")
#
# # ??????????????????????????????????????????
# # axes??????????????????????????????Yaxis?????????????????????????????????
# # ax1.yaxis.set_major_locator(MaxNLocator(nbins=5))
# # ax2.yaxis.set_major_locator(MaxNLocator(nbins=5))
#
# # ???????????????????????????????????????
# # axes??????????????????????????????Spine?????????????????????????????????
# # ??????????????????????????????ax2??????????????????
# ax2.spines['left'].set_color(color_1)
# ax2.spines['right'].set_color(color_2)
#
# # ???????????????????????????????????????
# ax1.tick_params(axis='y', colors=color_1)
# ax2.tick_params(axis='y', colors=color_2)
#
# # ??????????????????????????????????????????
# # ax1.yaxis.set_major_formatter(ticker.FormatStrFormatter("%d???"))
# # ax2.yaxis.set_major_formatter(ticker.FormatStrFormatter("%d???"))
#
# # ??????
# # ??????????????????????????????????????????????????????????????????????????????????????????barplot?????????plot??????????????????????????????
# handler1, label1 = ax1.get_legend_handles_labels()
# handler2, label2 = ax2.get_legend_handles_labels()
# # ?????????????????????????????????
# ax1.legend(handler1 + handler2, label1 + label2, loc=loc, borderaxespad=0., )
#
# ax1.set_ylim([0, 2000])
# ax2.set_ylim([0.97, 0.99])
# # ax1.axhline(y=433,linestyle='dashed', color=color_1)
# # ax2.axhline(y=0.980533,linestyle='dashed', color=color_2)
# ax2.axvline(x=max_test_acc_i+1, ymin=0, ymax=1, linestyle='dashed')
#
# plt.savefig(outpath)
# plt.show()