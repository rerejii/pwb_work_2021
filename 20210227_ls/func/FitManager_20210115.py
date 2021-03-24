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
        if restore:
            path_list = self.path_cls.search_newpoint_path(filename=self.path_cls.ckpt_epoch_name)
            if path_list:
                # tf.train.latest_checkpoint('./tf_estimator_example/')  # 別のチェックポイント指定方法
                root, ext = os.path.splitext(path_list[0])
                self.net_cls.ckpt_restore(path=root)
        total_step = self.net_cls.get_step()
        stock_time = []
        remain_step = ckpt_step - (total_step % ckpt_step)
        epoch = self.net_cls.get_epoch()
        shuf_train_ds_iter = self.shuf_train_ds_cls.get_inited_iter()
        while self.shuf_train_ds_cls.get_remain_data() is not 0:
            if epoch >= end_epoch:
                print('finish!')
                break  # fit関数終了
            # ===== 学習 =====
            start_time = time.time()
            pbar = tqdm(total=self.shuf_train_ds_cls.get_total_data(), desc='fitting (epoch ' + str(epoch + 1) + ')',
                        leave=False)
            pbar.update(self.shuf_train_ds_cls.get_total_data() - self.shuf_train_ds_cls.get_remain_data())
            remain_step = 200
            while remain_step > 0 and self.shuf_train_ds_cls.get_remain_data() != 0:
                # データ残数によってマルチデバイスかシングルデバイスかの判定を行う
                need_data_n = (self.shuf_train_ds_cls.get_batch_size() * len(device_list))
                if self.shuf_train_ds_cls.get_remain_data() >= need_data_n and remain_step >= need_data_n:
                    # マルチデバイス
                    err_count, c_weight = self.net_cls.multi_train_step(ds_iter=shuf_train_ds_iter, device_list=device_list)
                    self.shuf_train_ds_cls.data_used_apply(use_count=len(device_list))
                    progress_step = len(device_list) * self.shuf_train_ds_cls.get_batch_size()
                else:
                    # シングルデバイス
                    progress_step = self.shuf_train_ds_cls.get_next_data_n()
                    err_count, c_weight = self.net_cls.train_step(ds_iter=shuf_train_ds_iter, device_list=device_list,
                                            rate=self.shuf_train_ds_cls.get_next_data_rate())
                    self.shuf_train_ds_cls.data_used_apply(use_count=1)
                remain_step -= progress_step
                total_step += progress_step
                pbar.update(progress_step)  # プロセスバーを進行
                print(err_count)
                print(c_weight)
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
            if self.shuf_train_ds_cls.get_remain_data() == 0:
                print('saveing now!')
                epoch += 1
                self._epoch_proc(epoch, total_step, stock_time, latest_time)
                stock_time = []
                shuf_train_ds_iter = self.shuf_train_ds_cls.get_inited_iter()

    # ==============================================================================================================
    # ========== 保存関数 ==========================================================================================
    # ==============================================================================================================
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

