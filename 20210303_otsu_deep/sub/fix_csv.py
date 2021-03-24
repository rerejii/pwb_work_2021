import numpy as np
import pandas as pd
import csv
import os
import glob
import sys

nas_path = 'Z:/hayakawa/'
img_name = 'onestand'
directory_set = [
    'binary/20201218/unet_weight-d5-1_fin-none_logit-true_thr-d0',
    # 'binary/20201218/unet_fin-none_logit-true_thr-d5',
    # 'binary/20201218/unet_fin-sigmoid_logit-false_thr-d5',
    # 'binary/20201218/unet_fin-sigmoid_logit-true_thr-d5',
    # 'binary/20201218/unet_weight_fin-none_logit-true_thr-d0',
]
directory_set = [nas_path + directory_set[i] for i in range(len(directory_set))]

# directory_set = glob.glob('Z:/hayakawa/binary/20201218/*')
# print(directory_set)
# sys.exit()

# cols = ['train_gen_loss', 'test_gen_loss', 'valid_gen_loss']
# df = pd.read_csv('step_accuracy.csv')
# for col in cols:
#     for i in range(len(df)):
#         data = df[col][i]
#         if data[:9] == 'tf.Tensor':
#             find_i = data.find(',')
#             df[col][i] = float(data[10:find_i])


for dir_i in range(len(directory_set)):
    directory = directory_set[dir_i]
    csv_files = glob.glob(directory + '/*/CsvDatas/step_accuracy.csv')
    print(directory)
    if not csv_files:
        print('no file')
        continue
    for i in range(len(csv_files)):
        cols = ['train_gen_loss', 'test_gen_loss', 'valid_gen_loss']
        df = pd.read_csv(csv_files[i])
        for col in cols:
            for df_i in range(len(df)):
                data = df[col][df_i]
                if type(data) is str and data[:9] == 'tf.Tensor':
                    find_i = data.find(',')
                    df[col][df_i] = float(data[10:find_i])
        df.to_csv(csv_files[i])

# for col in cols:
#     for i in range(len(df)):
#         data = df[col][i]
#         print(data)

