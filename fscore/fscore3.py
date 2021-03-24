import pandas as pd
import numpy as np
import glob
import sys
from natsort import natsorted

def F1(precision, recall):
    out = (2*precision*recall) / (precision+recall)
    return out

def Fd5(precision, recall):  # 適合率重視
    out = (1.25*precision*recall) / (0.25*precision+recall)
    return out

def F2(precision, recall):  # 再現率重視
    out = (5*precision*recall) / (4*precision+recall)
    return out

root_path = 'Z:/hayakawa/binary/20210222/unet_use-bias_beta'
csv_path_set = glob.glob(root_path + '/**/CsvDatas/step_accuracy.csv')
csv_path_set = natsorted(csv_path_set)

for csv_i in range(len(csv_path_set)):
    csv_path = csv_path_set[csv_i]
    df = pd.read_csv(csv_path)
    df['test_N_F1'] = None
    df['test_N_Fd5'] = None
    df['test_N_F2'] = None
    df['validation_N_F1'] = None
    df['validation_N_Fd5'] = None
    df['validation_N_F2'] = None
    for df_i in range(len(df.index)):
        # test
        test_N_precision = df['test_N_precision'][df_i]
        test_N_recall = df['test_N_recall'][df_i]
        test_F1 = F1(test_N_precision, test_N_recall)
        test_Fd5 = Fd5(test_N_precision, test_N_recall)
        test_F2 = F2(test_N_precision, test_N_recall)
        df['test_N_F1'][df_i] = test_F1
        df['test_N_Fd5'][df_i] = test_Fd5
        df['test_N_F2'][df_i] = test_F2
        # validatin
        validation_N_precision = df['validation_N_precision'][df_i]
        validation_N_recall = df['validation_N_recall'][df_i]
        validation_F1 = F1(validation_N_precision, validation_N_recall)
        validation_Fd5 = Fd5(validation_N_precision, validation_N_recall)
        validation_F2 = F2(validation_N_precision, validation_N_recall)
        df['validation_N_F1'][df_i] = validation_F1
        df['validation_N_Fd5'][df_i] = validation_Fd5
        df['validation_N_F2'][df_i] = validation_F2
    # write
    df.to_csv(csv_path)
