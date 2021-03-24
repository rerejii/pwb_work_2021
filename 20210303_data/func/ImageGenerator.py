# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import math
import gc
import csv
import time as timefunc
import sys
import os
import pandas as pd
from tqdm import tqdm


def evenization(n):  # 偶数化
    return n + 1 if n % 2 != 0 else n


class ImageGenerator:
    def __init__(self, model_h, model_w, padding, Generator_model, fin_activate='sigmoid', use_gpu=True,
                 standardization_csv_path=None, hsv_to_rgp=False, standardization=False):
        self.model_h = model_h
        self.model_w = model_w
        self.padding = 64
        self.time_basis = 54000  # timeの基準
        self.day_time = 86400  # 1日
        self.fin_activate = fin_activate
        self.device = '/gpu:0' if use_gpu else '/cpu:0'
        self.Generator = tf.keras.models.load_model(Generator_model) if Generator_model is not None else None
        self.standardization_csv_path = standardization_csv_path
        self.std_list = None
        self.hsv_to_rgp = hsv_to_rgp
        self.standardization = standardization
        if self.standardization_csv_path is not None:
            df = pd.read_csv(self.standardization_csv_path)
            self.std_list = [ df['R_mean'] ,df['R_std'], df['G_mean'] ,df['G_std'], df['B_mean'] ,df['B_std'], ]
        print('Loading Generator')
        if use_gpu:
            with tf.device(self.device):
                zero_data = np.zeros(shape=[1, self.model_h, self.model_w, 3], dtype=np.float32)  # 空データ
                _ = self.generate_img(zero_data)  # GPUを使う場合,あらかじめGenerateを起動しておき,cudnnを先に呼び出しておく,と僅かに早くなる,気がする
        print('End Loading')

    # def chenge_generator(self, model):
    #     self.Generator = tf.keras.models.load_model(model)

    def binary_from_data(self, data, fin_activate):
        if fin_activate == 'tanh':
            return tf.greater_equal(data, 0)
        if fin_activate == 'sigmoid':
            return tf.greater_equal(data, 0.5)
        if fin_activate == 'None':
            return tf.greater_equal(data, 0)

    def img_from_netdata(self, data, fin_activate):
        if fin_activate == 'tanh':
            return (data + 1) * 127.5
        if fin_activate == 'sigmoid':
            return data * 255
        if fin_activate == 'None':
            return data * 255

    def netdata_from_img(self, data, fin_activate):
        if fin_activate == 'tanh':
            return (data / 127.5) - 1
        if fin_activate == 'sigmoid':
            return data / 255
        if fin_activate == 'None':
            return data / 255

    def img_standardization(self, data, std_list):
        data = tf.cast(data, tf.float32)
        data_R = (data[:, :, 0] - std_list[0]) / std_list[1]  # R_std, R_mean
        data_G = (data[:, :, 1] - std_list[2]) / std_list[3]
        data_B = (data[:, :, 2] - std_list[4]) / std_list[5]
        data = tf.stack([data_R, data_G, data_B], 2)
        return data

    def csv_standardization(self, data):
        data = tf.cast(data, tf.float32)
        data_R = (data[:, :, 0] - self.std_list[0]) / self.std_list[1]  # R_std, R_mean
        data_G = (data[:, :, 1] - self.std_list[2]) / self.std_list[3]
        data_B = (data[:, :, 2] - self.std_list[4]) / self.std_list[5]
        data = tf.stack([data_R, data_G, data_B], 2)
        return data

    def hsv_to_rgb(self, data):
        data = tf.cast(data, tf.float32)
        data = tf.image.rgb_to_hsv(data)
        return data

    def check_crop_index(self, norm_img):
        norm_h, norm_w, _ = norm_img.shape
        h_count = int((norm_h - self.padding * 2) / (self.model_h - (self.padding * 2)))
        w_count = int((norm_w - self.padding * 2) / (self.model_w - (self.padding * 2)))
        crop_h = [n // w_count for n in range(h_count * w_count)]
        crop_w = [n % w_count for n in range(h_count * w_count)]
        crop_top = [n * (self.model_h - (self.padding * 2)) for n in crop_h]
        crop_left = [n * (self.model_w - (self.padding * 2)) for n in crop_w]
        crop_index = list(zip(*[crop_top, crop_left]))
        return crop_index

    def _img_size_norm(self, img):
        origin_h, origin_w, _ = img.shape
        origin_h = float(origin_h)
        origin_w = float(origin_w)

        # 切り取る枚数を計算(切り取り枚数は偶数とする)
        sheets_h = evenization(math.ceil(origin_h / (self.model_h - self.padding * 2)))  # math.ceil 切り上げ
        sheets_w = evenization(math.ceil(origin_w / (self.model_w - self.padding * 2)))  # math.ceil 切り上げ

        # 必要となる画像サイズを求める(外側切り出しを考慮してpadding*2を足す)
        flame_h = sheets_h * (self.model_h - (self.padding * 2))  # 偶数 * N * 偶数 = 偶数
        flame_w = sheets_w * (self.model_w - (self.padding * 2))

        # 追加すべき画素数を求める
        extra_h = flame_h - origin_h  # if 偶数 - 奇数 = 奇数
        extra_w = flame_w - origin_w  # elif 偶数 - 偶数 = 偶数

        # print(flame_h)
        # print(flame_w)
        # sys.exit()

        # 必要画素数のフレームを作って中心に画像を挿入
        flame = np.zeros([flame_h, flame_w, 3], dtype=np.uint8)
        top, bottom = math.floor(extra_h / 2), flame_h - math.ceil(extra_h / 2)  # 追加画素が奇数なら下右側に追加させるceil
        left, right = math.floor(extra_w / 2), flame_w - math.ceil(extra_w / 2)  # 追加画素が奇数なら下右側に追加させるceil
        flame[top:bottom, left:right, :] = img
        return flame, [top, bottom, left, right]

    @tf.function  # 関数をグラフモードで実行 https://www.tensorflow.org/guide/autograph?hl=ja
    def generate_img(self, data):
        out = self.Generator(data, training=False)
        return self.binary_from_data(out, self.fin_activate)

    def get_str_time(self, time_num):
        ms_time, s_time = math.modf(time_num)  # ミニセカンド セカンド
        day, times = divmod(s_time, self.day_time)  # 日数と時間に
        day = int(day)
        step_times = timefunc.strptime(timefunc.ctime(times))
        str_time = str(day) + 'd' + timefunc.strftime("%H:%M:%S", step_times) + str(ms_time)[1:]
        return str_time

    def run(self, img_path, out_path, batch=50, time_out_path=None, ave_time_out_path=None, csv_reset=False):

        # プログラム稼働開始時刻取得
        time_stock = []
        time_stock.append(timefunc.time())

        # ========== 画像の読み込み read_time ==========
        print('--- Loading Image ---')
        img_byte = tf.io.read_file(img_path)
        in_img = tf.image.decode_png(img_byte, channels=3)
        time_stock.append(timefunc.time())  # 画像の読み込み終了時刻取得

        # ========== データセット作成 ds_time ==========
        height, width, _ = in_img.shape
        half_h = math.floor(height / 2)
        half_w = math.floor(width / 2)
        # cut_out_A = in_img[0: half_h, 0: half_w]
        # cut_out_B = in_img[0: half_h, half_w:]
        # cut_out_C = in_img[half_h: , 0: half_w]
        # cut_out_D = in_img[half_h: , half_w:]
        # in_img_list = [cut_out_A, cut_out_B, cut_out_C, cut_out_D]

        # for cut_img_index in range(4):
        # in_img = in_img_list[cut_img_index]
        print('--- Dataset Making ---')
        origin_h, origin_w, _ = in_img.shape  # 読み込み画像のサイズ取得
        in_img = np.array(in_img, dtype=np.uint8)  #
        norm_img, size = self._img_size_norm(in_img)  # 画像サイズを調整する

        # 切り出す画像の座標を求める
        crop_index = self.check_crop_index(norm_img)

        if self.standardization:
            R_data = np.array(in_img[:,:,0].ravel())
            G_data = np.array(in_img[:,:,1].ravel())
            B_data = np.array(in_img[:,:,2].ravel())
            R_std = np.std(R_data)
            R_mean = np.mean(R_data)
            G_std = np.std(G_data)
            G_mean = np.mean(G_data)
            B_std = np.std(B_data)
            B_mean = np.mean(B_data)
            std_list = [R_std, R_mean, G_std, G_mean, B_std, B_mean]

        # # 元画像を開放する
        # del in_img
        # gc.collect()

        # 画像切り出し用関数
        def _cut_img(img_index):
            cut = tf.slice(norm_img, [img_index[0], img_index[1], 0],
                           [self.model_h, self.model_w, 3])
            cut = tf.cast(cut, tf.float32)
            if self.standardization:
                cut = self.img_standardization(cut, std_list)
            elif self.standardization_csv_path is not None:
                cut = self.csv_standardization(cut)
            elif self.hsv_to_rgp:
                cut = self.hsv_to_rgb(cut)
            else:
                cut = self.netdata_from_img(cut, fin_activate=self.fin_activate)
            return cut

        # データセット
        dataset = tf.data.Dataset.from_tensor_slices(crop_index)
        # dataset = dataset.map(_cut_img)
        dataset = dataset.batch(batch)

        time_stock.append(timefunc.time())  # データセット定義終了時刻取得

        # ========== 画像生成 gen_time ==========
        print('--- Generate Image ---')
        norm_h, norm_w, _ = norm_img.shape
        out_flame = np.zeros(shape=[norm_h , norm_w, 1], dtype=np.float32)
        data_iter = iter(dataset)
        index = 0
        net_time = 0
        with tqdm(total=math.ceil(len(crop_index)), desc='image generate') as pbar:
            for data in data_iter:
                net_start_time = timefunc.time()
                with tf.device(self.device):
                    cut_data = []
                    for b in range(len(data)):
                        cut = _cut_img(data[b])
                        cut_data.append(cut)
                    cut_data = np.array(cut_data)
                    # cut = tf.expand_dims(cut, axis=0)
                    outset = self.generate_img(cut_data)
                net_time += timefunc.time() - net_start_time
                for out in outset:
                    crop_top, crop_left = crop_index[index]
                    crop_top = crop_top + self.padding
                    crop_left = crop_left + self.padding
                    out_flame[crop_top: crop_top + self.model_h - (self.padding * 2),
                    crop_left: crop_left + self.model_w - (self.padding * 2),
                    0] = out[self.padding: -1 * self.padding, self.padding: -1 * self.padding, 0]
                    index += 1
                pbar.update(batch)  # プロセスバーを進行
        out_img = self.img_from_netdata(out_flame, fin_activate=self.fin_activate)
        out_img = out_img[size[0]:size[1], size[2]:size[3]]
        time_stock.append(timefunc.time())  # 画像生成終了時刻取得

        # ========== 画像出力 out_time ==========
        print('--- Output Image ---')
        out_img = tf.cast(out_img, tf.uint8)
        out_byte = tf.image.encode_png(out_img)
        tf.io.write_file(filename=out_path, contents=out_byte)
        time_stock.append(timefunc.time())  # 画像出力終了時刻取得

        # ---------- 時間計測結果 ---------
        time_num_set = []
        time_str_set = []
        csv_header = ['read_time', 'ds_time', 'gen_time', 'out_time', 'total_time', 'only_net_time']
        for i in range(len(time_stock) - 1):
            time_num_set.append(time_stock[i + 1] - time_stock[i] + self.time_basis)
        time_num_set.append(time_stock[-1] - time_stock[0] + self.time_basis)
        time_num_set.append(net_time + self.time_basis)
        for i in range(len(time_num_set)):
            str_time = self.get_str_time(time_num_set[i])
            time_str_set.append(str_time)
            print(csv_header[i] + ': ' + str_time)

        if time_out_path is not None:
            if csv_reset and os.path.exists(time_out_path):
                os.remove(time_out_path)
            if not os.path.exists(time_out_path):
                with open(time_out_path, 'w') as f:
                    writer = csv.writer(f, lineterminator='\n')
                    writer.writerow(csv_header)
            with open(time_out_path, 'a') as f:
                writer = csv.writer(f, lineterminator='\n')
                writer.writerow(time_str_set)

        if ave_time_out_path is not None:
            # average
            if not os.path.exists(ave_time_out_path):
                csv_header = ['gen_time(str)', 'gen_time', 'gen_ave_time(str)', 'gen_ave_time']
                with open(ave_time_out_path, 'w') as f:
                    writer = csv.writer(f, lineterminator='\n')
                    writer.writerow(csv_header)
                    writer.writerow([time_str_set[2], time_num_set[2], time_str_set[2], time_num_set[2] ])
            else:
                with open(ave_time_out_path, 'a') as f:
                    writer = csv.writer(f, lineterminator='\n')
                    df = pd.read_csv(ave_time_out_path)
                    gen_times = df['gen_time'].values
                    gen_times = np.append(gen_times, time_num_set[2])
                    gen_ave_time = np.average(np.array(gen_times))
                    str_time = self.get_str_time(gen_ave_time)
                    writer.writerow([time_str_set[2], time_num_set[2], str_time, gen_ave_time])