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
from multiprocessing import Pool, Queue

'''
現在前提としてTEST_RATIO = VALID_RATIO = 1 とする (改善する必要があるまではこれで)
設計として、
1. 画像をいくつかのセットに等分割する(画像サイズは考慮しないものとする) data_split
分割数は LEARN_RATIO + TEST_RATIO + VALID_RATIO
2. TEST VALID をずらしながら作れるだけ作る
3. VALIDは一つ前のデータセットにてTESTにしたものを採用
'''

processes=4

CROP_H = 256
CROP_W = 256
PADDING = 0
RANDOM_SEED = 1
LEARN_RATIO = 4
TEST_RATIO = 1
VALID_RATIO = 1
SETNAME = [chr(i) for i in range(65,65+26)]  # A-Z
OUT_IMG_FOLDER = 'Img-256-valid'+str(PADDING)+'-more'
OUT_DS_FOLDER = 'DR-256-valid'+str(PADDING)+'-more'
OUT_PATH = 'E:/work/myTensor/dataset2/'

PROPERTY_FILE = 'Property'
LEARN_FILE = 'LearnData'
TEST_FILE = 'TestData'
VALID_FILE = 'ValidationData'
TEXT_FILE = 'TotalData'

origin_folder = 'E:/work/myTensor/dataset/img/*/*.png'
img_root_folder = OUT_PATH + OUT_IMG_FOLDER + '/'
ds_folder_root = OUT_PATH + OUT_DS_FOLDER
os.makedirs(ds_folder_root, exist_ok=True)


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
    sam_img = read_image(path=path, ch=3)
    # 解答画像 拡張子取って、+'_bin.png'
    ans_img = read_image(path=os.path.splitext(path)[0] + '_bin.png', ch=1)
    # 画像をうまく切り取れるようにリサイズする
    sam_norm, ans_norm = img_size_norm(sam_img, ans_img)
    # 画像の切り出し座標を求める
    crop_index = check_crop_index(sam_img)
    return sam_norm, ans_norm, crop_index


def queue_pre_proc(q_path):
    while True:
        # 画像の読み込み
        path = q_path.get()
        sam_img = read_image(path=path, ch=3)
        # 解答画像 拡張子取って、+'_bin.png'
        ans_img = read_image(path=os.path.splitext(path)[0] + '_bin.png', ch=1)
        # 画像をうまく切り取れるようにリサイズする
        sam_norm, ans_norm = img_size_norm(sam_img, ans_img)
        # 画像の切り出し座標を求める
        crop_index = check_crop_index(sam_img)
        # return sam_norm, ans_norm, crop_index


def make_recode(h, w, img_name, group_name, sample_path, answer_path):
    record = tf.train.Example(features=tf.train.Features(feature={
        'code_Y': tf.train.Feature(int64_list=tf.train.Int64List(value=[h])),
        'code_X': tf.train.Feature(int64_list=tf.train.Int64List(value=[w])),
        'height': tf.train.Feature(int64_list=tf.train.Int64List(value=[CROP_H])),
        'width': tf.train.Feature(int64_list=tf.train.Int64List(value=[CROP_W])),
        'padding': tf.train.Feature(int64_list=tf.train.Int64List(value=[PADDING])),
        'group_name': tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[group_name.encode()])),
        'img_name': tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[img_name.encode()])),
        'sample_path': tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[sample_path.encode()])),
        'answer_path': tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[answer_path.encode()])),
    }))
    return record


def make_property(csv_file, learn_count, test_count, valid_count, ):
    header = ['learn_image_height',
              'learn_image_width',
              'test_image_height',
              'test_image_width',
              'valid_image_height',
              'valid_image_width',
              'total_learn_data',
              'total_test_data',
              'total_valid_data',
              'padding_brank',
              'data_channel']
    val = [CROP_H,
           CROP_W,
           CROP_H,
           CROP_W,
           CROP_H,
           CROP_W,
           learn_count,
           test_count,
           valid_count,
           PADDING,
           3,]
    with open(csv_file, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerow(val)


def img_size_norm(sam_img, ans_img):
    # 切り取る枚数を計算(切り取り枚数は奇数・偶数関係なし 1枚全て学習、テストに回す)
    origin_h, origin_w, _ = sam_img.shape
    sheets_h = math.ceil(origin_h / CROP_H)  # math.ceil 切り上げ
    sheets_w = math.ceil(origin_w / CROP_W)  # math.ceil 切り上げ

    # 必要となる画像サイズを求める(外側切り出しを考慮してpadding*2を足す)
    flame_h = sheets_h * CROP_H + (PADDING * 2)
    flame_w = sheets_w * CROP_W + (PADDING * 2)

    # 追加すべき画素数を求める
    extra_h = flame_h - origin_h
    extra_w = flame_w - origin_w

    # 必要画素数のフレームを作って中心に画像を挿入
    sam_flame = np.zeros([flame_h, flame_w, 3], dtype=np.uint8)
    ans_flame = np.zeros([flame_h, flame_w, 1], dtype=np.uint8)
    top, bottom = math.floor(extra_h / 2), flame_h - math.ceil(extra_h / 2)  # 追加画素数が奇数なら下右側に追加させるceil
    left, right = math.floor(extra_w / 2), flame_w - math.ceil(extra_w / 2)  # 追加画素数が奇数なら下右側に追加させるceil
    sam_flame[top:bottom, left:right, :] = sam_img
    ans_flame[top:bottom, left:right] = ans_img
    return sam_flame, ans_flame


def check_crop_index(norm_img):
    norm_h, norm_w, _ = norm_img.shape
    h_count = int((norm_h - PADDING * 2) / CROP_H)
    w_count = int((norm_w - PADDING * 2) / CROP_W)
    crop_h = [n // w_count for n in range(h_count * w_count)]
    crop_w = [n % w_count for n in range(h_count * w_count)]
    crop_top = [n * CROP_H for n in crop_h]
    crop_left = [n * CROP_W for n in crop_w]
    crop_index = list(zip(*[crop_top, crop_left]))
    return crop_index


if __name__ == "__main__":

    # 乱数設定
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    os.environ['PYTHONHASHSEED'] = str(RANDOM_SEED)

    # 画像検索
    # in_img_path = [path for path in glob.glob(origin_folder) if path.find('_bin') is -1]
    in_img_path = [path for path in glob.glob(origin_folder) if len(os.path.basename(path)) is 9]
    print(in_img_path)

    # データセット分別
    ds_model_set = data_split(len(in_img_path), LEARN_RATIO, TEST_RATIO, VALID_RATIO)

    print(ds_model_set)
    sys.exit()


    # tfrecordのwirters生成
    # learn_ws, test_ws, valid_ws = make_tfrecord_writer(len(ds_model_set))

    # データカウント用
    total_learn_data = [0 for i in range(len(ds_model_set))]
    total_test_data = [0 for i in range(len(ds_model_set))]
    total_valid_data = [0 for i in range(len(ds_model_set))]

    def main_proc(crop_index_list, sam_norm, ans_norm, deep_norm, path_index):
        def crop_out(sam_out_path, ans_out_path, crop_top, crop_left):
            sam_crop = sam_norm[crop_top: crop_top + CROP_H + (PADDING * 2),
                       crop_left: crop_left + CROP_W + (PADDING * 2), :]
            sam_byte = tf.image.encode_png(sam_crop)
            tf.io.write_file(filename=sam_out_path, contents=sam_byte)
            ans_crop = ans_norm[crop_top: crop_top + CROP_H + (PADDING * 2),
                       crop_left: crop_left + CROP_H + (PADDING * 2)]
            ans_byte = tf.image.encode_png(ans_crop)
            tf.io.write_file(filename=ans_out_path, contents=ans_byte)

        def deep_crop_out(deep_out_path, crop_top, crop_left):
            deep_crop = deep_norm[crop_top: crop_top + CROP_H + (PADDING * 2),
                        crop_left: crop_left + CROP_W + (PADDING * 2), :]
            deep_byte = tf.image.encode_png(deep_crop)
            tf.io.write_file(filename=deep_out_path, contents=deep_byte)

        data_list = []
        with tqdm(total=len(crop_index_list), desc='TFRecord Processed') as pbar:
            for ci in range(len(crop_index_list)):
                crop_top, crop_left = crop_index_list[ci]
                # 画像プロパティ
                img_name = os.path.basename(os.path.splitext(in_img_path[path_index])[0])
                group_name = os.path.basename(os.path.dirname(in_img_path[path_index]))
                # 画像保存パス設定and生成
                sam_folder = img_root_folder + group_name + '/Sample/'
                ans_folder = img_root_folder + group_name + '/Answer/'
                os.makedirs(sam_folder, exist_ok=True)
                os.makedirs(ans_folder, exist_ok=True)
                h = int(crop_top / CROP_H)
                w = int(crop_left / CROP_W)
                str_Y = str(h).zfill(3)  # 左寄せゼロ埋め
                str_X = str(w).zfill(3)  # 左寄せゼロ埋め
                sam_out_path = sam_folder + '/Sample_X' + str_X + '_Y' + str_Y + '.png'
                ans_out_path = ans_folder + '/Answer_X' + str_X + '_Y' + str_Y + '.png'
                if not os.path.exists(sam_out_path) or not os.path.exists(ans_out_path):
                    data = [sam_out_path, ans_out_path, crop_top, crop_left]
                    data_list.append(data)
                sam_record_path = group_name + '/Sample' + '/Sample_X' + str_X + '_Y' + str_Y + '.png'
                ans_record_path = group_name + '/Answer' + '/Answer_X' + str_X + '_Y' + str_Y + '.png'
                recode = make_recode(h, w, img_name, group_name, sam_record_path, ans_record_path)
                for si in range(len(ds_model_set)):
                    test_si = si
                    valid_si = si - 1 if si is not 0 else len(ds_model_set) - 1
                    if path_index in ds_model_set[test_si]:
                        writer = test_ws[si]
                        total_test_data[si] += 1
                    elif path_index in ds_model_set[valid_si]:
                        writer = valid_ws[si]
                        total_valid_data[si] += 1
                    else:
                        writer = learn_ws[si]
                        total_learn_data[si] += 1
                    writer.write(recode.SerializeToString())
        with Pool(processes=processes) as pool:
            tqdm(pool.imap(crop_out, data_list), total=len(data_list), desc='Crop Out Processed')


    for si in range(len(ds_model_set)):
        csv_file = ds_folder_root + '/' + SETNAME[si] + '/' + PROPERTY_FILE + '-' + SETNAME[si] + '.csv'
        make_property(csv_file=csv_file,
                      learn_count=total_learn_data[si],
                      test_count=total_test_data[si],
                      valid_count=total_valid_data[si],)




















