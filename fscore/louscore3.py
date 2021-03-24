import pandas as pd
import numpy as np
import glob
import sys
from natsort import natsorted

def IoU(TN, FP, FN):
    out = TN / (TN + FP + FN)
    return out

root_path = 'Z:/hayakawa/binary/20210303/unet_rgb_ls'
csv_path_set = glob.glob(root_path + '/**/CsvDatas/eva_all.csv')
csv_path_set = natsorted(csv_path_set)

for csv_i in range(len(csv_path_set)):
    csv_path = csv_path_set[csv_i]
    df = pd.read_csv(csv_path)
    df['test_mean_IoU'] = None
    df['validation_mean_IoU'] = None
    for df_i in range(len(df.index)):
        # test
        test_IoU = df['test_IoU'][df_i]
        test_N_IoU = df['test_N_IoU'][df_i]
        test_mean_IoU = (test_IoU + test_N_IoU) / 2
        df['test_mean_IoU'][df_i] = test_mean_IoU
        # validation
        validation_IoU = df['validation_IoU'][df_i]
        validation_N_IoU = df['validation_N_IoU'][df_i]
        validation_mean_IoU = (validation_IoU + validation_N_IoU) / 2
        df['validation_mean_IoU'][df_i] = validation_mean_IoU
    df.to_csv(csv_path)
    print(csv_path)