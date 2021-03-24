import os
import glob
import csv
import sys
import numpy as np
import pandas as pd
from natsort import natsorted
import matplotlib.pyplot as plt

# csv_path = [
#     'unet_cw4-1-5_fin-none_logit-true_thr-d0/unet_cw4-1-5_fin-none_logit-true_thr-d0-A/CsvDatas/step_accuracy.csv',
#     'unet_cw4-1-10_fin-none_logit-true_thr-d0/unet_cw4-1-10_fin-none_logit-true_thr-d0-A/CsvDatas/step_accuracy.csv',
#     'unet_cw4-1-100_fin-none_logit-true_thr-d0/unet_cw4-1-100_fin-none_logit-true_thr-d0-A/CsvDatas/step_accuracy.csv',
#     'unet_cw4-d001-1_fin-none_logit-true_thr-d0/unet_cw4-d001-1_fin-none_logit-true_thr-d0-A/CsvDatas/step_accuracy.csv',
#     'unet_cw4-d01-1_fin-none_logit-true_thr-d0/unet_cw4-d01-1_fin-none_logit-true_thr-d0-A/CsvDatas/step_accuracy.csv',
#     'unet_cw4-d1-1_fin-none_logit-true_thr-d0/unet_cw4-d1-1_fin-none_logit-true_thr-d0-A/CsvDatas/step_accuracy.csv',
#     'unet_cw4-d5-1_fin-none_logit-true_thr-d0/unet_cw4-d5-1_fin-none_logit-true_thr-d0-A/CsvDatas/step_accuracy.csv',
#     'unet_cw5-d01-1_fin-none_logit-true_thr-d0/unet_cw5-d01-1_fin-none_logit-true_thr-d0-A/CsvDatas/step_accuracy.csv',
#     'unet_fin-none_logit-true_thr-d0/unet_fin-none_logit-true_thr-d0-A/CsvDatas/step_accuracy.csv',
# ]

root_path = 'Z:/hayakawa/binary/20210115/'
step_csv = 'step.csv'
check_path = [
    'unet_cw4-1-2_fin-none_logit-true_thr-d0/unet_cw4-1-2_fin-none_logit-true_thr-d0-A',
    'unet_cw4-1-5_fin-none_logit-true_thr-d0/unet_cw4-1-5_fin-none_logit-true_thr-d0-A',
    'unet_cw4-1-10_fin-none_logit-true_thr-d0/unet_cw4-1-10_fin-none_logit-true_thr-d0-A',
    'unet_cw4-1-100_fin-none_logit-true_thr-d0/unet_cw4-1-100_fin-none_logit-true_thr-d0-A',
    'unet_cw4-d001-1_fin-none_logit-true_thr-d0/unet_cw4-d001-1_fin-none_logit-true_thr-d0-A',
    'unet_cw4-d01-1_fin-none_logit-true_thr-d0/unet_cw4-d01-1_fin-none_logit-true_thr-d0-A',
    'unet_cw4-d1-1_fin-none_logit-true_thr-d0/unet_cw4-d1-1_fin-none_logit-true_thr-d0-A',
    'unet_cw4-d5-1_fin-none_logit-true_thr-d0/unet_cw4-d5-1_fin-none_logit-true_thr-d0-A',
    'unet_cw5-d01-1_fin-none_logit-true_thr-d0/unet_cw5-d01-1_fin-none_logit-true_thr-d0-A',
    'unet_cw6-1-2_fin-none_logit-true_thr-d0/unet_cw6-1-2_fin-none_logit-true_thr-d0-A',
    'unet_cw6-1-5_fin-none_logit-true_thr-d0/unet_cw6-1-5_fin-none_logit-true_thr-d0-A',
    'unet_fin-none_logit-true_thr-d0/unet_fin-none_logit-true_thr-d0-A',
    'unet_fin-none_logit-true_thr-d0_2/unet_fin-none_logit-true_thr-d0_2-A',
]
col_name_set = [
    '1-2',
    '1-5',
    '1-10',
    '1-100',
    '0.001-1',
    '0.01-1',
    '0.1-1',
    '0.5-1',
    'cw5 d01-1',
    'cw6 1-2',
    'cw6 1-5',
    'unet',
    'cw6 unet'
]
tag_name_set = [
    'test_accuracy',
    'validation_accuracy',
    'test_err_count',
    'validation_err_count',
    'test_ave_w',
    'validation_ave_w',
]

def get_step_list(ckpt_model_path, stock_model_path):
    train_step_list = []
    tmp = glob.glob(ckpt_model_path + '/' + 'ckpt-step*.index')
    ckpt_path_set = [os.path.splitext(t)[0] for t in tmp]
    tmp = glob.glob(stock_model_path + '/' + 'ckpt-step*.index')
    stock_path_set = [os.path.splitext(t)[0] for t in tmp]

    for i in range(len(ckpt_path_set)):
        ckpt_path = ckpt_path_set[i]
        start = ckpt_path.find('ckpt-step') + 9
        end = ckpt_path.find('-', start)
        total_step = int(ckpt_path[start: end])
        train_step_list.append(total_step)
    for i in range(len(stock_path_set)):
        ckpt_path = ckpt_path_set[i]
        start = ckpt_path.find('ckpt-step') + 9
        end = ckpt_path.find('-', start)
        total_step = int(ckpt_path[start: end])
        train_step_list.append(total_step)
        train_step_list = list(set(train_step_list))
        train_step_list = natsorted(train_step_list)
    return train_step_list


csv_path_set = [root_path + check_path[i] + '/CsvDatas/step_accuracy.csv' for i in range(len(check_path))]
ckpt_model_path_set = [root_path + check_path[i] + '/CheckpointModel/' for i in range(len(check_path))]
stock_model_path_set = [root_path + check_path[i] + '/StockModel/' for i in range(len(check_path))]
out_csv = root_path + 'plot/pic.csv'
out_df = pd.read_csv(step_csv, index_col=['step'])

for dir_i in range(len(check_path)):
    csv_path = csv_path_set[dir_i]
    ckpt_model_path = ckpt_model_path_set[dir_i]
    stock_model_path = stock_model_path_set[dir_i]
    train_step_list = get_step_list(ckpt_model_path, stock_model_path)

    # progress
    out_df[col_name_set[dir_i] + '_train_progress'] = None
    for step in train_step_list:
        out_df[col_name_set[dir_i] + '_train_progress'][step] = 'Trained'

    if os.path.isfile(csv_path):
        # print(csv_path)
        in_df = pd.read_csv(csv_path, index_col=['step'])
        # print(in_df)
        for tag_name in tag_name_set:
            # print(col_name_set[dir_i] + '_' + tag_name)
            out_df[col_name_set[dir_i] + '_' + tag_name] = in_df[tag_name]
    else:
        for tag_name in tag_name_set:
            # print(col_name_set[dir_i] + '_' + tag_name)
            out_df[col_name_set[dir_i] + '_' + tag_name] = None

out_df.to_csv(out_csv)





