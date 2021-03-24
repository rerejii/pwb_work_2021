import pandas as pd
import numpy as np
import glob
import sys
from natsort import natsorted

def IoU(TP, FP, FN):
    out = TP / (TP + FP + FN)
    return out

root_path = 'Z:/hayakawa/binary/20210303/unet_rgb_ls'
csv_path_set = glob.glob(root_path + '/**/CsvDatas/eva_all.csv')
csv_path_set = natsorted(csv_path_set)

for csv_i in range(len(csv_path_set)):
    csv_path = csv_path_set[csv_i]
    df = pd.read_csv(csv_path)
    df['test_IoU'] = None
    df['validation_IoU'] = None
    for df_i in range(len(df.index)):
        # test
        test_TP = df['test_TP'][df_i]
        test_FP = df['test_FP'][df_i]
        test_FN = df['test_FN'][df_i]
        test_IoU = IoU(TP=test_TP, FP=test_FP, FN=test_FN)
        df['test_IoU'][df_i] = test_IoU
        # validation
        validation_TP = df['validation_TP'][df_i]
        validation_FP = df['validation_FP'][df_i]
        validation_FN = df['validation_FN'][df_i]
        validation_IoU = IoU(TP=validation_TP, FP=validation_FP, FN=validation_FN)
        df['validation_IoU'][df_i] = validation_IoU
    df.to_csv(csv_path)
    print(csv_path)