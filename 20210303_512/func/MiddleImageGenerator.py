# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import math
import gc
import csv
import time as timefunc
import sys
import os
from tqdm import tqdm
from tensorflow.keras.utils import plot_model

def evenization(n):  # 偶数化
    return n + 1 if n % 2 != 0 else n

    # csvに書き込み
def write_csv(path, data):
    with open(path, 'a') as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerow(data)


class MiddleImageGenerator:
    def __init__(self, Generator_model, fin_activate='tanh', use_gpu=True):
        # self.model_h = model_h
        # self.model_w = model_w
        self.time_basis = 54000  # timeの基準
        self.day_time = 86400  # 1日
        self.fin_activate = fin_activate
        self.device = '/gpu:0' if use_gpu else '/cpu:0'
        self.Generator = tf.keras.models.load_model(Generator_model)
        print('Loading Generator')
        # if use_gpu:
        #     with tf.device(self.device):
        #         zero_data = np.zeros(shape=[1, self.model_h, self.model_w, 3], dtype=np.float32)  # 空データ
        #         _ = self.generate_img(zero_data)  # GPUを使う場合,あらかじめGenerateを起動しておき,cudnnを先に呼び出しておく,と僅かに早くなる,気がする
        print('End Loading')

    def netdata_from_img(self, data, fin_activate):
        if fin_activate is 'tanh':
            return (data / 127.5) - 1
        if fin_activate is 'sigmoid':
            return data / 255

    def run(self, img_path, out_path, layer_list, img_out=True, csv_out=True):

        # プログラム稼働開始時刻取得
        time_stock = []
        time_stock.append(timefunc.time())

        # ========== 画像の読み込み read_time ==========
        print('--- Loading Image ---')
        img_byte = tf.io.read_file(img_path)
        in_img = tf.image.decode_png(img_byte, channels=3)
        in_data = self.netdata_from_img(in_img, self.fin_activate)
        in_data = in_data[np.newaxis, :, :, :]
        # time_stock.append(timefunc.time())  # 画像の読み込み終了時刻取得

        # ==========  中間層出力の確認 ==========
        for layer_name in layer_list:
            hidden_layer_model = tf.keras.Model(
                inputs=self.Generator.input,
                outputs=self.Generator.get_layer(layer_name).output)
            hidden_out = hidden_layer_model(in_data, training=False)

            # ==========  画像の出力 ==========
            out_dir = out_path + '/' + layer_name
            csv_path = out_dir + '/' + layer_name+ '.csv'
            if csv_out and os.path.isfile(csv_path):
                os.remove(csv_path)
            os.makedirs(out_dir, exist_ok=True)
            for i in range(hidden_out.shape[3]):
                out_data = hidden_out[0, :, :, i]
                h, w = out_data.shape

                if img_out:
                    # sum_val = np.sun(out_data)
                    max_val = np.max(np.abs(out_data))
                    out_img = out_data / max_val * 255
                    out_img = tf.cast(out_img, tf.uint8)
                    # out_img = out_img.repeat(3). reshape(h,w,3)
                    out_img = tf.repeat(out_img, repeats=3)
                    out_img = tf.reshape(out_img, [h,w,3])
                    out_byte = tf.image.encode_png(out_img)
                    img_name = out_dir + '/' + layer_name + '_' + str(i) + '_' + str(max_val)+ '.png'
                    tf.io.write_file(filename=img_name, contents=out_byte)

                # ==========  csv出力 ==========
                if csv_out:
                    if not os.path.isfile(csv_path):
                        header_list = ['index | (h,w)',]
                        for hi in range(h):
                            for wi in range(w):
                                header_list.append('('+str(hi)+','+str(wi)+')')
                        header_list.append('max')
                        header_list.append('min')
                        write_csv(path=csv_path,data=header_list)
                    out_flat = tf.reshape(out_data, [-1])
                    max_val = np.max(out_data)
                    min_val = np.max(out_data)
                    csv_data = out_flat.numpy().tolist()
                    csv_data.insert(0, i)
                    csv_data.insert(-1, max_val)
                    csv_data.insert(-1, min_val)
                    write_csv(path=csv_path,data=csv_data)
    