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

root_path = 'Z:/hayakawa/binary/20210222/unet_use-bias_beta_otsu'
csv_path_set = glob.glob(root_path + '/**/CsvDatas/step_accuracy.csv')
csv_path_set = natsorted(csv_path_set)

for csv_i in range(len(csv_path_set)):
    csv_path = csv_path_set[csv_i]
    df = pd.read_csv(csv_path)
    df['test_F1'] = None
    df['test_Fd5'] = None
    df['test_F2'] = None
    df['validation_F1'] = None
    df['validation_Fd5'] = None
    df['validation_F2'] = None
    for df_i in range(len(df.index)):
        # test
        test_P_precision = df['test_P_precision'][df_i]
        test_P_recall = df['test_P_recall'][df_i]
        test_F1 = F1(test_P_precision, test_P_recall)
        test_Fd5 = Fd5(test_P_precision, test_P_recall)
        test_F2 = F2(test_P_precision, test_P_recall)
        df['test_F1'][df_i] = test_F1
        df['test_Fd5'][df_i] = test_Fd5
        df['test_F2'][df_i] = test_F2
        # validatin
        validation_P_precision = df['validation_P_precision'][df_i]
        validation_P_recall = df['validation_P_recall'][df_i]
        validation_F1 = F1(validation_P_precision, validation_P_recall)
        validation_Fd5 = Fd5(validation_P_precision, validation_P_recall)
        validation_F2 = F2(validation_P_precision, validation_P_recall)
        df['validation_F1'][df_i] = validation_F1
        df['validation_Fd5'][df_i] = validation_Fd5
        df['validation_F2'][df_i] = validation_F2
    # write
    df.to_csv(csv_path)
