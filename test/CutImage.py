import csv
import os
from tqdm import tqdm
import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
import sys
import math

def read_img(img_path):
    img_byte = tf.io.read_file(img_path)
    in_img = tf.image.decode_png(img_byte, channels=1)
    return tf.cast(tf.greater_equal(in_img, 127), tf.uint8) * 255

def output_img(out_img, out_path):
    out_byte = tf.image.encode_png(out_img)
    tf.io.write_file(filename=out_path, contents=out_byte)

def evenization(n):  # 偶数化
    return n + 1 if n % 2 != 0 else n

padding = 32
model_h = 128
model_w = 128
def check_crop_index(norm_img):
    norm_h, norm_w, _ = norm_img.shape

    h_count = int((norm_h - padding * 2) / model_h)
    w_count = int((norm_w - padding * 2) / model_w)
    crop_h = [n // w_count for n in range(h_count * w_count)]
    crop_w = [n % w_count for n in range(h_count * w_count)]
    crop_top = [n * model_h for n in crop_h]
    crop_left = [n * model_w for n in crop_w]
    crop_index = list(zip(*[crop_top, crop_left]))
    return crop_index

def _img_size_norm(img):
    origin_h, origin_w, _ = img.shape

    # 切り取る枚数を計算(切り取り枚数は偶数とする)
    sheets_h = math.ceil(origin_h / (model_h - padding))  # math.ceil 切り上げ
    sheets_w = math.ceil(origin_w / (model_w - padding))  # math.ceil 切り上げ
    print(sheets_h)
    print(sheets_w)

    # 必要となる画像サイズを求める(外側切り出しを考慮してpadding*2を足す)
    flame_h = sheets_h * model_h + (padding * 2)  # 偶数 * N * 偶数 = 偶数
    flame_w = sheets_w * model_w + (padding * 2)

    # 追加すべき画素数を求める
    extra_h = flame_h - origin_h  # if 偶数 - 奇数 = 奇数
    extra_w = flame_w - origin_w  # elif 偶数 - 偶数 = 偶数

    # 必要画素数のフレームを作って中心に画像を挿入
    flame = np.zeros([flame_h, flame_w, 3], dtype=np.uint8)
    top, bottom = math.floor(extra_h / 2), flame_h - math.ceil(extra_h / 2)  # 追加画素が奇数なら下右側に追加させるceil
    left, right = math.floor(extra_w / 2), flame_w - math.ceil(extra_w / 2)  # 追加画素が奇数なら下右側に追加させるceil
    flame[top:bottom, left:right, :] = img
    return flame, [top, bottom, left, right]

# def _cut_img()

# 128*128の画像を3*3=9枚生成することを目的

ans_path = 'E:/work/myTensor/dataset2/Img-256-gray-valid0-more/17H-0863-1_L0001/Answer/Answer_X021_Y029.png'
ans_img = read_img(ans_path)
flame, index = _img_size_norm(ans_img)
print(flame.shape)
# sys.exit()
crop_index = check_crop_index(flame)
print(crop_index)
