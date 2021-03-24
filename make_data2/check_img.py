import os
import numpy as np
import cv2
import sys
import math
import random
from tqdm import tqdm
from check_brank import check_brank
import tensorflow as tf
import csv
import glob

'''
現在前提としてTEST_RATIO = VALID_RATIO = 1 とする (改善する必要があるまではこれで)
設計として、
1. 画像をいくつかのセットに等分割する(画像サイズは考慮しないものとする) data_split
分割数は LEARN_RATIO + TEST_RATIO + VALID_RATIO
2. TEST VALID をずらしながら作れるだけ作る
3. VALIDは一つ前のデータセットにてTESTにしたものを採用
'''

CROP_H = 256
CROP_W = 256
PADDING = 0
out_path_root = 'E:/work/myTensor/dataset2/crop/unet_L0017/'
image_path = 'D:/Users/hayakawa/Desktop/leo2_unet_01_sigmoid_more/out_L0017.png'
# out_path_root = 'E:/work/myTensor/dataset2/crop/encode_L0017/'
# image_path = 'D:/Users/hayakawa/Desktop/leo1_encode_0001_more/out_L0017.png'
def read_image(path, ch):
    byte = tf.io.read_file(path)
    data = tf.image.decode_png(byte, channels=ch)
    return np.array(data, dtype=np.uint8)

def check_crop_index(norm_img):
    norm_h, norm_w, _ = norm_img.shape
    h_count = int(norm_h / CROP_H)
    w_count = int(norm_w / CROP_W)
    crop_h = [n // w_count for n in range(h_count * w_count)]
    crop_w = [n % w_count for n in range(h_count * w_count)]
    crop_top = [n * CROP_H for n in crop_h]
    crop_left = [n * CROP_W for n in crop_w]
    crop_index = list(zip(*[crop_top, crop_left]))
    return crop_index

def pre_proc(path):
    img = read_image(path=path, ch=1)
    # 画像をうまく切り取れるようにリサイズする
    img_norm = img_size_norm(img)
    # 画像の切り出し座標を求める
    crop_index = check_crop_index(img_norm)
    return img_norm, crop_index

def img_size_norm(img):
    # 切り取る枚数を計算(切り取り枚数は奇数・偶数関係なし 1枚全て学習、テストに回す)
    # print(img.shape)
    origin_h, origin_w, _ = img.shape
    sheets_h = math.ceil(origin_h / CROP_H)  # math.ceil 切り上げ
    sheets_w = math.ceil(origin_w / CROP_W)  # math.ceil 切り上げ

    # 必要となる画像サイズを求める(外側切り出しを考慮してpadding*2を足す)
    flame_h = sheets_h * CROP_H + (PADDING * 2)
    flame_w = sheets_w * CROP_W + (PADDING * 2)

    # 追加すべき画素数を求める
    extra_h = flame_h - origin_h
    extra_w = flame_w - origin_w

    # 必要画素数のフレームを作って中心に画像を挿入
    # sam_flame = np.zeros([flame_h, flame_w, 3], dtype=np.uint8)
    flame = np.zeros([flame_h, flame_w, 1], dtype=np.uint8)
    top, bottom = math.floor(extra_h / 2), flame_h - math.ceil(extra_h / 2)  # 追加画素数が奇数なら下右側に追加させるceil
    left, right = math.floor(extra_w / 2), flame_w - math.ceil(extra_w / 2)  # 追加画素数が奇数なら下右側に追加させるceil
    # sam_flame[top:bottom, left:right, :] = sam_img
    flame[top:bottom, left:right] = img
    return flame


img_norm, crop_index = pre_proc(image_path)
os.makedirs(out_path_root, exist_ok=True)

with tqdm(total=len(crop_index), desc='Processed') as pbar:
    for ci in range(len(crop_index)):
        crop_top, crop_left = crop_index[ci]
        h = int(crop_top / CROP_H)
        w = int(crop_left / CROP_W)
        img_crop = img_norm[crop_top: crop_top + CROP_H + (PADDING * 2),
                            crop_left: crop_left + CROP_H + (PADDING * 2)]
        str_Y = str(h).zfill(3)  # 左寄せゼロ埋め
        str_X = str(w).zfill(3)  # 左寄せゼロ埋め
        out_path = out_path_root + '/Output_X' + str_X + '_Y' + str_Y + '.png'
        if not os.path.exists(out_path):
            img_byte = tf.image.encode_png(img_crop)
            tf.io.write_file(filename=out_path, contents=img_byte)
        pbar.update(1)  # プロセスバーを進行

