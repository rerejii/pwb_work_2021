import os
import numpy as np
import sys
import math
import random
from tqdm import tqdm
import csv
import glob
import pandas as pd
import shutil

# ========== GPU設定 ==========
args = sys.argv
DEVICE_NUMBER_STR = '0'  # 使用するGPU設定
# device番号のlistを生成(複数GPU学習時の割り当て調整に使用)
DEVICE_STR_LIST = DEVICE_NUMBER_STR.split(',')
# DEVICE_LIST = [int(n) for n in DEVICE_STR_LIST]
DEVICE_LIST = [int(n) for n in range(len(DEVICE_STR_LIST))]
# 環境変数の調整
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = DEVICE_NUMBER_STR

import tensorflow as tf

CROP_H = 256
CROP_W = 256
PADDING = 128
RANDOM_SEED = 1
LEARN_RATIO = 4
TEST_RATIO = 1
VALID_RATIO = 1
SETNAME = [chr(i) for i in range(65, 65+26)]  # A-Z
OUT_IMG_FOLDER = 'Img-256-valid'+str(PADDING)+'-more'

# OUT_PATH = 'E:/work/myTensor/dataset2/'
# OUT_PATH = '/home/hayakawa/work/myTensor/dataset2/'
OUT_PATH = '/localhome/rerejii/work/myTensor/dataset2/'
# OUT_PATH = 'C:/Users/hayakawa/work/mytensor/dataset2/'
# OUT_PATH = 'Z:/hayakawa/work/myTensor/dataset2/'
# OUT_PATH = '/nas-homes/krlabmember/hayakawa/work/myTensor/dataset2/'
IN_PATH = '/nas-homes/krlabmember/hayakawa/work20/dataset2/'
# IN_PATH = 'Z:/hayakawa/work20/dataset2/'

# ans_folder = IN_PATH + 'img/17H-0863-1_L0012/L0012_bin.png'
# boundary_folder = IN_PATH + 'img/17H-0863-1_L0012/L0012_bin_boundary.png'
img_root_folder = OUT_PATH + OUT_IMG_FOLDER + '/'

# 乱数設定
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
os.environ['PYTHONHASHSEED'] = str(RANDOM_SEED)


def evenization(n):  # 偶数化
    return n + 1 if n % 2 != 0 else n


def read_image(path, ch):
    byte = tf.io.read_file(path)
    data = tf.image.decode_png(byte, channels=ch)
    return np.array(data, dtype=np.uint8)

def pre_proc(path, ch):
    img = read_image(path=path, ch=ch)
    origin_h, origin_w = img.shape[:2]
    sheets_h = math.ceil(origin_h / CROP_H)  # math.ceil 切り上げ
    sheets_w = math.ceil(origin_w / CROP_W)  # math.ceil 切り上げ

    # 必要となる画像サイズを求める(外側切り出しを考慮してpadding*2を足す)
    flame_h = sheets_h * CROP_H + (PADDING * 2)
    flame_w = sheets_w * CROP_W + (PADDING * 2)

    # 追加すべき画素数を求める
    extra_h = flame_h - origin_h
    extra_w = flame_w - origin_w

    # 必要画素数のフレームを作って中心に画像を挿入
    flame = np.zeros([flame_h, flame_w, 1], dtype=np.uint8)
    top, bottom = math.floor(extra_h / 2), flame_h - math.ceil(extra_h / 2)  # 追加画素数が奇数なら下右側に追加させるceil
    left, right = math.floor(extra_w / 2), flame_w - math.ceil(extra_w / 2)  # 追加画素数が奇数なら下右側に追加させるceil
    flame[top:bottom, left:right] = img
    return flame

def check_crop_index():
    # norm_h, norm_w, _ = norm_img.shape
    norm_h, norm_w = [19608, 24576]
    h_count = int(norm_h / CROP_H)
    w_count = int(norm_w / CROP_W)
    crop_h = [n // w_count for n in range(h_count * w_count)]
    crop_w = [n % w_count for n in range(h_count * w_count)]
    crop_top = [n * CROP_H for n in crop_h]
    crop_left = [n * CROP_W for n in crop_w]
    crop_index = list(zip(*[crop_top, crop_left]))
    return crop_index

ans_path = IN_PATH + 'img/17H-0863-1_L0012/L0012_bin.png'
boundary_path = IN_PATH + 'img/17H-0863-1_L0012/L0012_bin_boundary.png'

ans_norm = pre_proc(ans_path, ch=1)
boundary_norm = pre_proc(boundary_path, ch=1)
crop_index = check_crop_index()
group_name = '17H-0863-1_L0012'
ans_folder = img_root_folder + group_name + '/Answer/'
boundary_folder = img_root_folder + group_name + '/Boundary/'

os.makedirs(ans_folder, exist_ok=True)
os.makedirs(boundary_folder, exist_ok=True)

with tqdm(total=len(crop_index), desc='Processed') as pbar:
    for ci in range(len(crop_index)):
        crop_top, crop_left = crop_index[ci]
        h = int(crop_top / CROP_H)
        w = int(crop_left / CROP_W)
        ans_crop = ans_norm[crop_top: crop_top + CROP_H + (PADDING * 2),
                   crop_left: crop_left + CROP_H + (PADDING * 2)]
        boundary_crop = boundary_norm[crop_top: crop_top + CROP_H + (PADDING * 2),
                   crop_left: crop_left + CROP_H + (PADDING * 2)]
        str_Y = str(h).zfill(3)  # 左寄せゼロ埋め
        str_X = str(w).zfill(3)  # 左寄せゼロ埋め
        ans_out_path = ans_folder + '/Answer_X' + str_X + '_Y' + str_Y + '.png'
        boundary_out_path = boundary_folder + '/Boundary_X' + str_X + '_Y' + str_Y + '.png'
        ans_byte = tf.image.encode_png(ans_crop)
        tf.io.write_file(filename=ans_out_path, contents=ans_byte)
        boundary_byte = tf.image.encode_png(boundary_crop)
        tf.io.write_file(filename=boundary_out_path, contents=boundary_byte)
        pbar.update(1)  # プロセスバーを進行
