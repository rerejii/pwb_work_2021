# import tensorflow as tf
import numpy as np
import cv2
import csv
from natsort import natsorted
import glob
import pandas as pd
import sys
from tqdm import tqdm

# path
IMAGE_DIR = 'Z:/hayakawa/work20/dataset2/'
SAM_PATH = 'Z:/hayakawa/binary/20210303/unet_rgb_otsu_loop'
CSV_PATH = 'Z:/hayakawa/binary/20210303/unet_rgb/rgb_acc_otsu_loop.csv'

# 画像
ANS_IMAGE_SET = ['img/17H-0863-1_L0001/L0001_bin.png',
             'img/17H-0863-1_L0002-old/L0002_bin.png',
             'img/17H-0863-1_L0003/L0003_bin.png',
             'img/17H-0863-1_L0004/L0004_bin.png',
             'img/17H-0863-1_L0005/L0005_bin.png',
             'img/17H-0863-1_L0006-new/L0006_bin.png',
             'img/17H-0863-1_L0007/L0007_bin.png',
             'img/17H-0863-1_L0008/L0008_bin.png',
             'img/17H-0863-1_L0009/L0009_bin.png',
             'img/17H-0863-1_L0010/L0010_bin.png',
             'img/17H-0863-1_L0011/L0011_bin.png',
             'img/17H-0863-1_L0012/L0012_bin.png',
             'img/17H-0863-1_L0013/L0013_bin.png',
             'img/17H-0863-1_L0014/L0014_bin.png',
             'img/17H-0863-1_L0015/L0015_bin.png',
             'img/17H-0863-1_L0016/L0016_bin.png',
             'img/17H-0863-1_L0017/L0017_bin.png',
             'img/17H-0863-1_L0018/L0018_bin.png',]

# pathの合成
c = [IMAGE_DIR + '/' + s for s in ANS_IMAGE_SET]

def get_IoU(TP, FP, FN):
    out = TP / (TP + FP + FN)
    return out

def get_IoU_N(TN, FP, FN):
    out = TN / (TN + FP + FN)
    return out

# 混同行列の評価
def evaluate_confusion_matrix(output, label):
    one = 255
    zero = 0
    TP = np.sum( np.array( np.logical_and( np.equal(output, one), np.equal(label, one) ), np.float32 ) )
    FP = np.sum( np.array( np.logical_and( np.equal(output, one), np.equal(label, zero) ), np.float32 ) )
    FN = np.sum( np.array( np.logical_and( np.equal(output, zero), np.equal(label, one) ), np.float32 ) )
    TN = np.sum( np.array( np.logical_and( np.equal(output, zero), np.equal(label, zero) ), np.float32 ) )
    return [TP, FP, FN, TN]

def evaluate(out, ans):
    out_binary = out
    ans_binary = ans
    correct_prediction = np.equal(out_binary, ans_binary)
    confusion_matrix = evaluate_confusion_matrix(out_binary, ans_binary)
    return np.mean(np.array(correct_prediction, np.float32)), confusion_matrix

df = pd.DataFrame(None,columns=['img_name', 'accuracy', 'mean_IoU'],)
df = df.set_index('img_name')
total_acc = 0.
total_mean_IoU = 0.
for i in tqdm(range(18), desc='Processed'):
    s = str(i + 1).zfill(2)
    df.loc['generator_L00' + s + '.png'] = None
    SAM_IMAGE = glob.glob(SAM_PATH + '/**/generator_L00' + s + '.png')[0]
    ANS_IMAGE = ANS_IMAGE_SET[i]
    gen_output = cv2.imread(SAM_IMAGE, 0)
    target = cv2.imread(ANS_IMAGE, 0)

    acc, confusion_matrix = evaluate(out=gen_output, ans=target)
    TP, FP, FN, TN = confusion_matrix
    # print(TN)
    IoU = get_IoU(TP=TP, FP=FP, FN=FN)
    IoU_N = get_IoU_N(TN=TN, FP=FP, FN=FN)
    mean_IoU = (IoU + IoU_N) / 2
    print(acc)
    print(mean_IoU)
    df.loc['generator_L00' + s + '.png'] = [acc, mean_IoU]
    total_acc += acc
    total_mean_IoU += mean_IoU
# print(df)
df.to_csv(CSV_PATH)