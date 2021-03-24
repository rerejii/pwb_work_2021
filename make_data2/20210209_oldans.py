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
RANDOM_SEED = 1
LEARN_RATIO = 4
TEST_RATIO = 1
VALID_RATIO = 1
SETNAME = [chr(i) for i in range(65, 65+26)]  # A-Z
OUT_IMG_FOLDER = 'Img-256-valid'+str(PADDING)+'-old_ans'
# OUT_DS_FOLDER = 'DR-256-20210208-valid'+str(PADDING)+'-more'

# OUT_PATH = 'E:/work/myTensor/dataset2/'
# OUT_PATH = '/home/hayakawa/work/myTensor/dataset2/'
# OUT_PATH = '/localhome/rerejii/work/myTensor/dataset2/'
OUT_PATH = 'C:/Users/hayakawa/work/mytensor/dataset2/'
# OUT_PATH = 'Z:/hayakawa/work/myTensor/dataset2/'
# OUT_PATH = '/nas-homes/krlabmember/hayakawa/work/myTensor/dataset2/'
# IN_PATH = '/nas-homes/krlabmember/hayakawa/work20/dataset2/'
IN_PATH = 'Z:/hayakawa/work20/dataset2/'

PROPERTY_FILE = 'Property'
LEARN_FILE = 'LearnData'
TEST_FILE = 'TestData'
VALID_FILE = 'ValidationData'
TEXT_FILE = 'TotalData'
origin_folder = IN_PATH + 'img/17H-0863-1_L0012/*.png'
stand_csv_path = 'one_std.csv'
# print(origin_folder)
# origin_folder = 'E:/work/myTensor/dataset2/img/*/*.png'
# origin_folder = '/home/hayakawa/work/myTensor/dataset2/img/*/*.png'
# origin_folder = 'C:/Users/hayakawa/work/mytensor/dataset2/img/*/*.png'

img_root_folder = OUT_PATH + OUT_IMG_FOLDER + '/'
# ds_folder_root = OUT_PATH + OUT_DS_FOLDER
# os.makedirs(ds_folder_root, exist_ok=True)

def evenization(n):  # 偶数化
    return n + 1 if n % 2 != 0 else n


def read_image(path, ch):
    byte = tf.io.read_file(path)
    data = tf.image.decode_png(byte, channels=ch)
    return np.array(data, dtype=np.uint8)


def data_split(data_n, learn_ratio, test_ratio, valid_ratio):
    if data_n % (learn_ratio + test_ratio + valid_ratio) is not 0:
        print('データセットとデータ分別比の合計が割り切れないよ！')
    ran_list = random.sample(list(range(data_n)), data_n)
    one_ds = data_n // (learn_ratio + test_ratio + test_ratio)  # int
    ds_model_set = []
    for i in range(int(data_n / one_ds)):
        # [[0, 1, 2], [8, 13, 15], [5, 1, 0], [3, 12, 6], [10, 14, 4], [16, 9, 2], [7, 11, 17]]
        ds_model_set.append(ran_list[i * one_ds: i * one_ds + one_ds])
    print(ds_model_set)  # ============ デバック用出力 ============
    return ds_model_set


def make_tfrecord_writer(set_n):
    learn_ws = []
    test_ws = []
    valid_ws = []
    for si in range(set_n):
        # フォルダ生成
        ds_folder = ds_folder_root + '/' + SETNAME[si]
        os.makedirs(ds_folder, exist_ok=True)
        # ファイル名設定
        learn_tf = ds_folder + '/' + LEARN_FILE + '-' + SETNAME[si] + '.tfrecords'
        test_tf = ds_folder + '/' + TEST_FILE + '-' + SETNAME[si] + '.tfrecords'
        valid_tf = ds_folder + '/' + VALID_FILE + '-' + SETNAME[si] + '.tfrecords'
        # tfrecordのwirters生成
        learn_ws.append(tf.io.TFRecordWriter(learn_tf))
        test_ws.append(tf.io.TFRecordWriter(test_tf))
        valid_ws.append(tf.io.TFRecordWriter(valid_tf))
    return learn_ws, test_ws, valid_ws


def pre_proc(path):
    # 画像の読み込み
    # sam_img = read_image(path=path, ch=3)
    # 解答画像 拡張子取って、+'_bin.png'
    ans_img = read_image(path=os.path.splitext(path)[0] + '_bin_old.png', ch=1)
    # # deep画像 拡張子取って、+'_deep_19.png'
    # weight_img = read_image(path=os.path.splitext(path)[0] + '_bin_morphology-weight_21.png', ch=1)
    # # 距離画像
    # dis_img = read_image(path=os.path.splitext(path)[0] + '_bin_dst.png', ch=1)
    # # 境界画像
    # bound_img = read_image(path=os.path.splitext(path)[0] + '_bin_boundary.png', ch=1)
    # # hls画像
    # hls_img = read_image(path=os.path.splitext(path)[0] + '_hls.png', ch=3)
    # # brank_otsu_l画像
    # brank_otsu_l_img = read_image(path=os.path.splitext(path)[0] + '_brank_otsu_l.png', ch=1)
    # brank_otsu_l_c3_img = read_image(path=os.path.splitext(path)[0] + '_brank_otsu_l_c3.png', ch=1)
    # brank_otsu_l_c5_img = read_image(path=os.path.splitext(path)[0] + '_brank_otsu_l_c5.png', ch=1)
    #
    # otsu127_l_img = read_image(path=os.path.splitext(path)[0] + '_brank_otsu_l127.png', ch=1)
    # otsu127_s_img = read_image(path=os.path.splitext(path)[0] + '_otsu127_s.png', ch=1)
    # 画像をうまく切り取れるようにリサイズする
    ans_norm = img_size_norm(ans_img)
    # 画像の切り出し座標を求める
    crop_index = check_crop_index(ans_img)
    return ans_norm, crop_index

def img_size_norm(ans_img):
    # 切り取る枚数を計算(切り取り枚数は奇数・偶数関係なし 1枚全て学習、テストに回す)
    origin_h, origin_w, _ = ans_img.shape
    sheets_h = math.ceil(origin_h / CROP_H)  # math.ceil 切り上げ
    sheets_w = math.ceil(origin_w / CROP_W)  # math.ceil 切り上げ

    # 必要となる画像サイズを求める(外側切り出しを考慮してpadding*2を足す)
    flame_h = sheets_h * CROP_H + (PADDING * 2)
    flame_w = sheets_w * CROP_W + (PADDING * 2)

    # 追加すべき画素数を求める
    extra_h = flame_h - origin_h
    extra_w = flame_w - origin_w

    # 必要画素数のフレームを作って中心に画像を挿入
    ans_flame = np.zeros([flame_h, flame_w, 1], dtype=np.uint8)
    top, bottom = math.floor(extra_h / 2), flame_h - math.ceil(extra_h / 2)  # 追加画素数が奇数なら下右側に追加させるceil
    left, right = math.floor(extra_w / 2), flame_w - math.ceil(extra_w / 2)  # 追加画素数が奇数なら下右側に追加させるceil
    ans_flame[top:bottom, left:right] = ans_img
    return ans_flame


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


# 乱数設定
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
os.environ['PYTHONHASHSEED'] = str(RANDOM_SEED)

# 画像検索
# in_img_path = [path for path in glob.glob(origin_folder) if path.find('_bin') is -1]
in_img_path = [path for path in glob.glob(origin_folder) if len(os.path.basename(path)) is 9]

# データセット分別
# ds_model_set = data_split(len(in_img_path), LEARN_RATIO, TEST_RATIO, VALID_RATIO)

# tfrecordのwirters生成
# learn_ws, test_ws, valid_ws = make_tfrecord_writer(len(ds_model_set))

# データカウント用
# total_learn_data = [0 for i in range(len(ds_model_set))]
# total_test_data = [0 for i in range(len(ds_model_set))]
# total_valid_data = [0 for i in range(len(ds_model_set))]

df_stand = pd.read_csv(stand_csv_path, index_col=0)

for pi in range(len(in_img_path)):
    print('Image %d / %d' % (pi + 1, len(in_img_path)))
    # if in_img_path[pi][-9:] == 'L0001.png':
    #     print('check')
    #     continue
    # 画像関連の前処理

    ans_norm, crop_index = pre_proc(in_img_path[pi])
    # print(ans_norm.shape)
    # sys.exit()

    # 画像プロパティ
    img_name = os.path.basename(os.path.splitext(in_img_path[pi])[0])

    # print(df_stand.at[img_name, 'R_std'])
    # print(df_stand.loc[img_name].values)
    # sys.exit()

    group_name = os.path.basename(os.path.dirname(in_img_path[pi]))
    # 画像保存パス設定and生成
    ans_folder = img_root_folder + group_name + '/OldAnswer/'

    os.makedirs(ans_folder, exist_ok=True)

    # print(len(crop_index))
    # sys.exit()

    with tqdm(total=len(crop_index), desc='Processed') as pbar:
        # 画像の切り出し
        for ci in range(len(crop_index)):
            crop_top, crop_left = crop_index[ci]
            h = int(crop_top / CROP_H)
            w = int(crop_left / CROP_W)
            ans_crop = ans_norm[crop_top: crop_top + CROP_H + (PADDING * 2),
                       crop_left: crop_left + CROP_H + (PADDING * 2)]
            # deep_crop = deep_norm[crop_top: crop_top + CROP_H + (PADDING * 2),
            #             crop_left: crop_left + CROP_W + (PADDING * 2), :]
            str_Y = str(h).zfill(3)  # 左寄せゼロ埋め
            str_X = str(w).zfill(3)  # 左寄せゼロ埋め
            ans_out_path = ans_folder + '/Answer_X' + str_X + '_Y' + str_Y + '.png'

            ans_record_path = group_name + '/Answer' + '/Answer_X' + str_X + '_Y' + str_Y + '.png'


            ans_byte = tf.image.encode_png(ans_crop)
            tf.io.write_file(filename=ans_out_path, contents=ans_byte)

            pbar.update(1)  # プロセスバーを進行

# for si in range(len(ds_model_set)):
#     csv_file = ds_folder_root + '/' + SETNAME[si] + '/' + PROPERTY_FILE + '-' + SETNAME[si] + '.csv'
#     make_property(csv_file, total_learn_data[si], total_test_data[si], total_valid_data[si])
#     print('===================================')
#     print(SETNAME[si] + '-learn_data: ' + str(total_learn_data[si]))
#     print(SETNAME[si] + '-test_data: ' + str(total_test_data[si]))
#     print(SETNAME[si] + '-valid_data: ' + str(total_valid_data[si]))