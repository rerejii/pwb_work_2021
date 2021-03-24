import glob
import numpy as np
import os
import sys
from tqdm import tqdm
args = sys.argv
DEVICE_NUMBER_STR = args[1]  # 使用するGPU設定
DEVICE_STR_LIST = DEVICE_NUMBER_STR.split(',')
DEVICE_LIST = [int(n) for n in range(len(DEVICE_STR_LIST))]
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = DEVICE_NUMBER_STR


import tensorflow as tf

# # 評価画像の出力
boundary_kernel = 5
crush_kernel = 9
black = [0, 0, 0]  # 予測:0黒 解答:0黒
white = [255, 255, 255]  # 予測:1白 解答:1白
red = [255, 0, 0]  # 予測:1白 解答:0黒
blue = [0, 204, 255]  # 予測:0黒 解答:1白

img_path_root = 'Z:/hayakawa/share/'
img_path = [
    'img/17H-0863-1_L0006-new/L0006_bin.png',
    'img/17H-0863-1_L0011/L0011_bin.png',
    'img/17H-0863-1_L0017/L0017_bin.png',
    'img/17H-0863-1_L0003/L0003_bin.png',
    'img/17H-0863-1_L0005/L0005_bin.png',
    'img/17H-0863-1_L0009/L0009_bin.png',
    'img/17H-0863-1_L0002-old/L0002_bin.png',
    'img/17H-0863-1_L0008/L0008_bin.png',
    'img/17H-0863-1_L0013/L0013_bin.png',
    'img/17H-0863-1_L0007/L0007_bin.png',
    'img/17H-0863-1_L0012/L0012_bin.png',
    'img/17H-0863-1_L0014/L0014_bin.png',
    'img/17H-0863-1_L0004/L0004_bin.png',
    'img/17H-0863-1_L0015/L0015_bin.png',
    'img/17H-0863-1_L0016/L0016_bin.png',
    'img/17H-0863-1_L0001/L0001_bin.png',
    'img/17H-0863-1_L0010/L0010_bin.png',
    'img/17H-0863-1_L0018/L0018_bin.png',
]
img_path = [img_path_root + path for path in img_path]

def read_img(img_path):
    img_byte = tf.io.read_file(img_path)
    in_img = tf.image.decode_png(img_byte, channels=1)
    return tf.cast(tf.greater_equal(in_img, 127), tf.uint8) * 255

def out_img(out_img, out_path):
    out_img = tf.cast(out_img, tf.uint8)
    out_byte = tf.image.encode_png(out_img)
    tf.io.write_file(filename=out_path, contents=out_byte)

def erosion(img, kernel_size=3):
    img = tf.expand_dims(img, axis=0)  # バッチ次元付与
    kernel = np.ones((kernel_size, kernel_size, 1), np.float32)
    img = tf.cast(img, tf.float32)
    img = tf.nn.erosion2d(value=img, filters=kernel, strides=[1, 1, 1, 1], padding='SAME',
                          data_format='NHWC', dilations=[1, 1, 1, 1]) + 1  # 白収縮 何故か出力が-1される
    img = tf.cast(tf.greater_equal(img, 127), tf.uint8) * 255
    img = img[0, :, :, :]
    return img

def dilation(img, kernel_size=3):
    img = tf.expand_dims(img, axis=0)  # バッチ次元付与
    kernel = np.ones((kernel_size, kernel_size, 1), np.float32)
    img = tf.cast(img, tf.float32)
    img = tf.nn.dilation2d(input=img, filters=kernel, strides=[1, 1, 1, 1], padding='SAME',
                          data_format='NHWC', dilations=[1, 1, 1, 1]) - 1  # 白拡大 何故か出力が+1される
    img = tf.cast(tf.greater_equal(img, 127), tf.uint8) * 255
    img = img[0, :, :, :]
    return img

def binary_to_img(img):
    return tf.greater_equal(img, 127)

def boundary(in_img, kernel_size=5):
    # weight = tf.zeros_like(in_img)
    img = in_img
    ero_img = erosion(img, kernel_size=kernel_size)
    dil_img = dilation(img, kernel_size=kernel_size)

    img_b = binary_to_img(img)
    ero_img = binary_to_img(ero_img)
    dil_img = binary_to_img(dil_img)

    weight = tf.cast(tf.not_equal(img_b, ero_img), tf.uint8) + tf.cast(tf.not_equal(img_b, dil_img), tf.uint8)
    return weight

def check_crush(in_img, kernel_size=5):
    img = in_img
    ero_img = erosion(img, kernel_size=kernel_size)
    ero_crush_img = dilation(ero_img, kernel_size=kernel_size)
    dil_img = dilation(img, kernel_size=kernel_size)
    dil_crush_img = erosion(dil_img, kernel_size=kernel_size)

    img_b = binary_to_img(img)
    ero_crush_img = binary_to_img(ero_crush_img)
    dil_crush_img = binary_to_img(dil_crush_img)

    weight = tf.cast(tf.not_equal(img_b, ero_crush_img), tf.uint8) + tf.cast(tf.not_equal(img_b, dil_crush_img), tf.uint8)
    return weight

ans_binary_list = img_path
pbar = tqdm(total=len(ans_binary_list))
for img_idx in range(len(ans_binary_list)):
    img = read_img(ans_binary_list[img_idx])
    img_name = os.path.splitext(os.path.basename(ans_binary_list[img_idx]))[0]
    h, w = img.numpy().shape[:2]
    weight = boundary(img, kernel_size=boundary_kernel)
    crush = check_crush(img, kernel_size=crush_kernel)
    result_img = np.zeros([h, w, 3])
    img_3d = img.numpy().repeat(3).reshape(h, w, 3) / 255
    w_3d = weight.numpy().repeat(3).reshape(h, w, 3)
    c_3d = crush.numpy().repeat(3).reshape(h, w, 3)
    result_img += (w_3d == 0) * (img_3d == 0) * black
    result_img += (w_3d == 0) * (img_3d == 1) * white
    result_img += (w_3d == 1) * red
    out_img(result_img, 'crop_img/out/'+img_name+'_k'+str(boundary_kernel)+'.png')

    result_img = np.zeros([h, w, 3])
    result_img += (w_3d == 0) * (img_3d == 0) * black
    result_img += (w_3d == 0) * (img_3d == 1) * white
    result_img += (c_3d == 1) * (img_3d == 0) * black
    result_img += (c_3d == 1) * (img_3d == 1) * white
    result_img += (w_3d == 1) * (c_3d == 0) * red
    out_img(result_img, 'crop_img/out/' + img_name + '_bk' + str(boundary_kernel) + '_ck' + str(crush_kernel) + '.png')
    pbar.update(1)  # プロセスバーを進行
pbar.close()  # プロセスバーの終了



