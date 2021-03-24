import pandas as pd
import numpy as np

df = pd.read_csv('A.csv')

def F1(precision, recall):
    out = (2*precision*recall) / (precision+recall)
    return out

def Fd5(precision, recall):  # 適合率重視
    out = (1.25*precision*recall) / (0.25*precision+recall)
    return out

def F2(precision, recall):  # 再現率重視
    out = (5*precision*recall) / (4*precision+recall)
    return out

# F1
df['test_F1'] = None
df['test_Fd5'] = None
df['test_F2'] = None
df['validation_F1'] = None
df['validation_Fd5'] = None
df['validation_F2'] = None

print(df.loc[0])
for i in range(len(df.index)):
    # test
    test_P_precision = df['test_P_precision'][i]
    test_P_recall = df['test_P_recall'][i]
    test_F1 = F1(test_P_precision, test_P_recall)
    test_Fd5 = Fd5(test_P_precision, test_P_recall)
    test_F2 = F2(test_P_precision, test_P_recall)
    df['test_F1'][i] = test_F1
    df['test_Fd5'][i] = test_Fd5
    df['test_F2'][i] = test_F2
    # validatin
    validation_P_precision = df['validation_P_precision'][i]
    validation_P_recall = df['validation_P_recall'][i]
    validation_F1 = F1(validation_P_precision, validation_P_recall)
    validation_Fd5 = Fd5(validation_P_precision, validation_P_recall)
    validation_F2 = F2(validation_P_precision, validation_P_recall)
    df['validation_F1'][i] = validation_F1
    df['validation_Fd5'][i] = validation_Fd5
    df['validation_F2'][i] = validation_F2
