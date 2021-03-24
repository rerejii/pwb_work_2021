# -*- coding: utf-8 -*-

import tensorflow as tf
from tqdm import tqdm
# from tqdm import tqdm_notebook as tqdm
import shutil
import time
import os
import numpy as np
import ShareNetFunc as nfunc
import math
import sys
import csv
import pandas as pd

""" 
混同行列
    1   0 (予測)
1   TP  FN 
0   FP  TN
(実際)
TP : True Positive
TN : True Negative
FP : False Positive
FN : False Negative
適合率/精度(precision) = TP/(FP+TP)　陽性であると予測した内の何%が当たっていたかを示す。
再現率(recall) = TP/(FN+TP)  本当に陽性であるケースの内、何％を陽性と判断できたかを示す。

P_precision = TP/(FP+TP)
N_Negative = TN/(FN+TN)
P_recall = TP/(FN+TP)
N_recall = TN/(FP+TN)
"""


class FitManager():
    def __init__(self,
                 net_cls,
                 path_cls,
                 shuf_train_ds_cls,
                 train_ds_cls,
                 test_ds_cls,
                 validation_ds_cls=None,
                 check_img_path=[], ):
        self.net_cls = net_cls
        self.path_cls = path_cls
        self.shuf_train_ds_cls = shuf_train_ds_cls
        self.train_ds_cls = train_ds_cls
        self.test_ds_cls = test_ds_cls
        self.validation_ds_cls = validation_ds_cls
        self.step_writer = tf.summary.create_file_writer(logdir=self.path_cls.get_step_log_path())
        self.check_img_path = check_img_path
        self.time_csv_header = ['step', 'study_time', 'total_time', 'test_time(ms)']
        self.summary_image_tag = ['learn', 'test', 'valid']
        self.small_region = 148
        self.middle_region = 210
        self.large_region = 256
        self.time_basis = 54000  # timeの基準
        self.day_time = 86400  # 1日

    # ==============================================================================================================
    # ========== 外部呼出し関数 =====================================================================================
    # ==============================================================================================================
    # epoch単位のデータでシャッフルを行う関係上、step毎にデータをストックしてepoch終了時に一括で正式に保存する方式
    def fit(self,
            end_epoch,
            device_list=[0],
            ckpt_step=None,
            restore=True, ):
        if ckpt_step is None:
            ckpt_step = self.shuf_train_ds_cls.get_total_data()
        if not os.path.isfile(self.path_cls.make_csv_path(filename=self.path_cls.time_csv_name)):
            nfunc.write_csv(path_cls=self.path_cls, filename=self.path_cls.time_csv_name,
                            datalist=[self.time_csv_header])
        print(self.path_cls.make_csv_path('step_loss.csv'))
        if not os.path.isfile(self.path_cls.make_csv_path('step_loss.csv')):
            with open(self.path_cls.make_csv_path('step_loss.csv'), 'w') as f:
                writer = csv.writer(f, lineterminator='\n')
                writer.writerow(['step', 'loss'])
        if not os.path.isfile(self.path_cls.make_csv_path('train_loss.csv')):
            with open(self.path_cls.make_csv_path('train_loss.csv'), 'w') as f:
                writer = csv.writer(f, lineterminator='\n')
                writer.writerow(['step', 'loss'])
        if not os.path.isfile(self.path_cls.make_csv_path('train_data.csv')):
            with open(self.path_cls.make_csv_path('train_data.csv'), 'w') as f:
                writer = csv.writer(f, lineterminator='\n')
                writer.writerow(['epoch', 'total_err_count', 'total_c_weight', 'loss'])


        if restore:
            path_list = self.path_cls.search_newpoint_path(filename=self.path_cls.ckpt_epoch_name)
            if path_list:
                # tf.train.latest_checkpoint('./tf_estimator_example/')  # 別のチェックポイント指定方法
                root, ext = os.path.splitext(path_list[0])
                self.net_cls.ckpt_restore(path=root)

        # self.net_cls.ckpt_restore(path='/nas-homes/krlabmember/hayakawa/work20/model/ckpt-step350208-22')

        total_step = self.net_cls.get_step()
        stock_time = []
        remain_step = ckpt_step - (total_step % ckpt_step)
        epoch = self.net_cls.get_epoch()
        shuf_train_ds_iter = self.shuf_train_ds_cls.get_inited_iter()
        epoch_loss = 0.
        ckpt_loss = 0.
        ckpt_count_step = 0
        total_count_step = 0
        while self.shuf_train_ds_cls.get_remain_data() is not 0:
            if epoch >= end_epoch:
                print('finish!')
                break  # fit関数終了
            # ===== 学習 =====
            start_time = time.time()
            pbar = tqdm(total=self.shuf_train_ds_cls.get_total_data(), desc='fitting (epoch ' + str(epoch + 1) + ')',
                        leave=False)
            pbar.update(self.shuf_train_ds_cls.get_total_data() - self.shuf_train_ds_cls.get_remain_data())
            total_err_count = 0
            total_c_weight = 0

            while remain_step > 0 and self.shuf_train_ds_cls.get_remain_data() != 0:
                # データ残数によってマルチデバイスかシングルデバイスかの判定を行う
                need_data_n = (self.shuf_train_ds_cls.get_batch_size() * len(device_list))
                data_set = next(shuf_train_ds_iter)
                ds = data_set[0:4]
                path = data_set[-1]
                start_time = time.time()
                if self.shuf_train_ds_cls.get_remain_data() >= need_data_n and remain_step >= need_data_n:
                    # マルチデバイス
                    gen_output, err_count, c_weight, train_loss = self.net_cls.multi_train_step(ds=ds, device_list=device_list)
                    self.shuf_train_ds_cls.data_used_apply(use_count=len(device_list))
                    progress_step = len(device_list) * self.shuf_train_ds_cls.get_batch_size()
                else:
                    # シングルデバイス
                    progress_step = self.shuf_train_ds_cls.get_next_data_n()
                    gen_output, err_count, c_weight, train_loss = self.net_cls.train_step(ds=ds, device_list=device_list,
                                            rate=self.shuf_train_ds_cls.get_next_data_rate())
                    self.shuf_train_ds_cls.data_used_apply(use_count=1)

                end_time = time.time()
                ms_time, s_time = math.modf((end_time - start_time) + self.time_basis)  # ミニセカンド セカンド
                day, times = divmod(s_time, self.day_time)  # 日数と時間に
                day = int(day)
                step_times = time.strptime(time.ctime(times))
                str_time = str(day) + ':' + time.strftime("%H:%M:%S", step_times) + str(ms_time)[1:]
                # print('train time is : ' + str_time)
                train_loss = train_loss.numpy()
                # self.weight_image(data_set, gen_output, c_weight)
                remain_step -= progress_step
                total_step += progress_step
                total_err_count += err_count.numpy()
                total_c_weight += tf.reduce_sum(c_weight).numpy()
                epoch_loss += train_loss
                ckpt_loss += train_loss
                ckpt_count_step += 1
                total_count_step += 1
                print('loss: ' +str(train_loss))
                # print(self.net_cls.gen_optimizer.learning_rate)
                # df = pd.read_csv(self.path_cls.make_csv_path('step_loss.csv'), index_col=0)
                # df.loc[total_step] = [train_loss]
                # df.to_csv(self.path_cls.make_csv_path('step_loss.csv'))
                pbar.update(progress_step)  # プロセスバーを進行
            pbar.close()  # プロセスバーの終了
            self.net_cls.add_study_time(time.time() - start_time)
            # ===== 保存 =====
            study_time_str = self.net_cls.get_str_study_time()
            total_time_str = self.net_cls.get_str_total_time()
            latest_time = [total_step, study_time_str, total_time_str]
            if remain_step <= 0:
                self._step_proc(total_step)
                stock_time.append(latest_time)
                remain_step = remain_step + ckpt_step
                w_loss = ckpt_loss / ckpt_count_step
                df = pd.read_csv(self.path_cls.make_csv_path('train_loss.csv'), index_col=0)
                df.loc[total_step] = [w_loss]
                df.to_csv(self.path_cls.make_csv_path('train_loss.csv'))
                ckpt_count_step = 0
                ckpt_loss = 0.
            if self.shuf_train_ds_cls.get_remain_data() == 0:
            # if True:
                print('saveing now!')
                epoch += 1
                self._epoch_proc(epoch, total_step, stock_time, latest_time)
                stock_time = []
                shuf_train_ds_iter = self.shuf_train_ds_cls.get_inited_iter()
                with open(self.path_cls.make_csv_path('train_data.csv'), 'a') as f:
                    writer = csv.writer(f, lineterminator='\n')
                    w_loss = epoch_loss / total_count_step
                    writer.writerow([epoch, w_loss])
                    epoch_loss = 0.
                    total_count_step = 0

    # ==============================================================================================================
    # ========== 保存関数 ==========================================================================================
    # ==============================================================================================================
    def weight_image(self, ds, gen_output_batch, c_weight):
        black = [0, 0, 0]  # 予測:0黒 解答:0黒
        white = [255, 255, 255]  # 予測:1白 解答:1白
        red = [255, 0, 0]  # 予測:1白 解答:0黒
        blue = [0, 204, 255]  # 予測:0黒 解答:1白
        root_path = self.path_cls.output_rootfolder
        input_image_batch, target_batch, weight_batch, path_batch = ds
        gen = self.net_cls.binary_from_data(gen_output_batch, label='output')
        gen = tf.cast(gen, tf.uint8) * 255
        for i in range(input_image_batch.get_shape()[0]):
            s = path_batch[i].numpy().decode()
            w_path = root_path + '/WeightImage/' + s
            o_path = root_path + '/OutputImage/' + s[:-20] + 'Output' + s[-14:]
            os.makedirs(os.path.dirname(w_path), exist_ok=True)
            result_img = np.zeros([256, 256, 3])
            out_3d = gen[i].numpy().repeat(3).reshape(256, 256, 3) / 255
            # result_img = out_3d
            ans_3d = target_batch[i].numpy().repeat(3).reshape(256, 256, 3)
            w_3d = c_weight[i].numpy().repeat(3).reshape(256, 256, 3)
            # print(w_3d)
            result_img += (w_3d == 0) * (ans_3d == 0) * black  # 黒 正解
            result_img += (ans_3d == 1) * (out_3d == 1) * (w_3d == 0) * white  # 白 正解
            result_img += (w_3d == 1) * red  # 赤 黒欠け
            result_img += (out_3d == 0) * (ans_3d == 1) * (w_3d == 0) * blue
            result_img += (out_3d == 1) * (ans_3d == 0) * (w_3d == 0) * blue
            result_img = tf.cast(result_img, tf.uint8)
            out_byte = tf.image.encode_png(result_img)
            tf.io.write_file(filename=w_path, contents=out_byte)
            out_byte = tf.image.encode_png(gen[i])
            tf.io.write_file(filename=o_path, contents=out_byte)

    def _step_proc(self, total_step):
        # ckptのstep値更新
        self.net_cls.set_ckpt_val(step_val=total_step)
        # stock
        stock_path = self.path_cls.make_stock_path(filename=self.path_cls.ckpt_step_name + str(total_step))
        self.net_cls.get_checkpoint().save(stock_path)
        # 画像の保存
        nfunc.img_check(step=total_step, net_cls=self.net_cls,
                        path_cls=self.path_cls, check_img_path=self.check_img_path)

    def _epoch_proc(self, epoch, total_step, stock_time, latest_time):
        latest_time_li = [latest_time]
        # save
        self.net_cls.set_ckpt_val(step_val=total_step, epoch_val=epoch)
        for stock_ckpt in self.path_cls.search_stock_path(filename=self.path_cls.ckpt_step_name):
            shutil.move(stock_ckpt, self.path_cls.make_checkpoint_path(filename=''))
        # epoch単位のckpt(復元用)
        old_path = self.path_cls.search_newpoint_path(filename=self.path_cls.ckpt_epoch_name)
        epoch_path = self.path_cls.make_newpoint_path(filename=self.path_cls.ckpt_epoch_name + str(epoch))
        self.net_cls.get_checkpoint().save(epoch_path)
        for path in old_path:
            os.remove(path)
        # ログの書き込み
        nfunc.write_csv(path_cls=self.path_cls, filename=self.path_cls.time_csv_name, datalist=stock_time)
        nfunc.write_csv(path_cls=self.path_cls, filename=self.path_cls.epoch_time_csv_name, datalist=latest_time_li)

