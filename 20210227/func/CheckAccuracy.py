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
from natsort import natsorted
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
                 train_ds_cls,
                 test_ds_cls,
                 validation_ds_cls=None,
                 check_img_path=[], ):
        self.net_cls = net_cls
        self.path_cls = path_cls
        self.train_ds_cls = train_ds_cls
        self.test_ds_cls = test_ds_cls
        self.validation_ds_cls = validation_ds_cls
        self.step_writer = tf.summary.create_file_writer(logdir=self.path_cls.get_step_log_path())
        self.check_img_path = check_img_path
        self.accuracy_csv_header = ['step', 'train_accuracy', 'test_accuracy', 'validation_accuracy',
                                    'train_P_precision', 'train_N_precision',
                                    'train_P_recall', 'train_N_recall',
                                    'test_P_precision', 'test_N_precision',
                                    'test_P_recall', 'test_N_recall',
                                    'validation_P_precision', 'validation_N_precision',
                                    'validation_P_recall', 'validation_N_recall',
                                    'train_TP', 'train_FP', 'train_FN', 'train_TN',
                                    'test_TP', 'test_FP', 'test_FN', 'test_TN',
                                    'validation_TP', 'validation_FP', 'validation_FN', 'validation_TN',
                                    'train_small_region', 'train_middle_region', 'train_large_region',
                                    'test_small_region', 'test_middle_region', 'test_large_region',
                                    'valid_small_region', 'valid_middle_region', 'valid_large_region',
                                    'train_gen_loss',
                                    'test_gen_loss',
                                    'valid_gen_loss',
                                    'test_time']
        self.time_csv_header = ['step', 'study_time', 'total_time', 'test_time(ms)']
        self.summary_scalar_name = ['train_accuracy', 'test_accuracy', 'validation_accuracy',
                                    'train_P_precision', 'train_N_precision',
                                    'train_P_recall', 'train_N_recall',
                                    'test_P_precision', 'test_N_precision',
                                    'test_P_recall', 'test_N_recall',
                                    'validation_P_precision', 'validation_N_precision',
                                    'validation_P_recall', 'validation_N_recall',
                                    'train_TP', 'train_FP', 'train_FN', 'train_TN',
                                    'test_TP', 'test_FP', 'test_FN', 'test_TN',
                                    'validation_TP', 'validation_FP', 'validation_FN', 'validation_TN',
                                    'train_small_region', 'train_middle_region', 'train_large_region',
                                    'train_small_region', 'train_middle_region', 'train_large_region',
                                    'test_small_region', 'test_middle_region', 'test_large_region',
                                    'valid_small_region', 'valid_middle_region', 'valid_large_region',
                                    'train_gen_loss',
                                    'test_gen_loss',
                                    'valid_gen_loss',]
        self.summary_image_tag = ['learn', 'test', 'valid']
        self.small_region = 148
        self.middle_region = 210
        self.large_region = 256

    # ==============================================================================================================
    # ========== 外部呼出し関数 =====================================================================================
    # ==============================================================================================================
    # epoch単位のデータでシャッフルを行う関係上、step毎にデータをストックしてepoch終了時に一括で正式に保存する方式
    def run(self,
            end_epoch,
            device_list=[0],
            ckpt_step=None,
            restore=True, ):

        # csvファイルの作成
        csv_file = self.path_cls.make_csv_path('acc_ckpt.csv')
        if not os.path.isfile(csv_file):
            with open(csv_file, 'w') as f:
                writer_total = csv.writer(f, lineterminator='\n')
                writer_total.writerow(self.accuracy_csv_header)

        df = pd.read_csv(csv_file)

        # checkpoint の検索
        ckpt_path_set = self.path_cls.search_checkpoint_path()
        ckpt_path_set = natsorted(ckpt_path_set)

        for i in range(len(ckpt_path_set)):
            ckpt_path = ckpt_path_set[i]
            start = ckpt_path.find('ckpt-step') + 9
            end = ckpt_path.find('-', start)
            total_step = int(ckpt_path[start: end])

            if total_step in df['step'].values:
                continue

            epoch = total_step // ckpt_step
            root, ext = os.path.splitext(ckpt_path[0])
            self.net_cls.ckpt_restore(path=root)

            # ===== 評価 =====
            print('==== start ckpt ' + str(total_step) + '====')
            train_accuracy, train_confusion_matrix, train_region_acc, train_gen_loss = self.check_accuracy(
                ds_cls=self.train_ds_cls,
                device_list=device_list,
                task_name='train_ds check (epoch:' + str(epoch + 1) + ' step:' + str(total_step) + ')'
            )
            train_P_precision = train_confusion_matrix[0] / (train_confusion_matrix[1] + train_confusion_matrix[0])
            train_N_precision = train_confusion_matrix[3] / (train_confusion_matrix[2] + train_confusion_matrix[3])
            train_P_recall = train_confusion_matrix[0] / (train_confusion_matrix[2] + train_confusion_matrix[0])
            train_N_recall = train_confusion_matrix[3] / (train_confusion_matrix[1] + train_confusion_matrix[3])

            test_accuracy, test_confusion_matrix, test_region_acc, test_gen_loss = self.check_accuracy(
                ds_cls=self.test_ds_cls,
                device_list=device_list,
                task_name='test_ds check (epoch:' + str(epoch + 1) + ' step:' + str(total_step) + ')'
            )
            test_P_precision = test_confusion_matrix[0] / (test_confusion_matrix[1] + test_confusion_matrix[0])
            test_N_precision = test_confusion_matrix[3] / (test_confusion_matrix[2] + test_confusion_matrix[3])
            test_P_recall = test_confusion_matrix[0] / (test_confusion_matrix[2] + test_confusion_matrix[0])
            test_N_recall = test_confusion_matrix[3] / (test_confusion_matrix[1] + test_confusion_matrix[3])

            if self.validation_ds_cls is not None:
                validation_accuracy, validation_confusion_matrix, validation_region_acc, validation_gen_loss = self.check_accuracy(
                    ds_cls=self.validation_ds_cls,
                    device_list=device_list,
                    task_name='validation_ds check (epoch:' + str(epoch + 1) + ' step:' + str(total_step) + ')'
                )
            else:
                validation_accuracy = test_accuracy
                validation_confusion_matrix = test_confusion_matrix
                validation_region_acc = [0.0, 0.0, 0.0]
                validation_gen_loss = test_gen_loss
            validation_P_precision = validation_confusion_matrix[0] / (
                    validation_confusion_matrix[1] + validation_confusion_matrix[0])
            validation_N_precision = validation_confusion_matrix[3] / (
                    validation_confusion_matrix[2] + validation_confusion_matrix[3])
            validation_P_recall = validation_confusion_matrix[0] / (
                    validation_confusion_matrix[2] + validation_confusion_matrix[0])
            validation_N_recall = validation_confusion_matrix[3] / (
                    validation_confusion_matrix[1] + validation_confusion_matrix[3])
            print(str(epoch + 1) + 'epoch ' + str(total_step) + 'step accuracy [train, test, valid]: ['
                  + str(train_accuracy) + ', ' + str(test_accuracy) + ', ' + str(validation_accuracy) + ']')
            print(str(epoch + 1) + 'epoch ' + str(
                total_step) + 'step train [P_precision, N_precision, P_recall, N_recall]: ['
                  + str(train_P_precision) + ', ' + str(train_N_precision) + ', '
                  + str(train_P_recall) + ', ' + str(train_N_recall) + ', ' + ']')
            print(str(epoch + 1) + 'epoch ' + str(
                total_step) + 'step test [P_precision, N_precision, P_recall, N_recall]: ['
                  + str(test_P_precision) + ', ' + str(test_N_precision) + ', '
                  + str(test_P_recall) + ', ' + str(test_N_recall) + ', ' + ']')
            print(str(epoch + 1) + 'epoch ' + str(
                total_step) + 'step validation [P_precision, N_precision, P_recall, N_recall]: ['
                  + str(validation_P_precision) + ', ' + str(validation_N_precision) + ', '
                  + str(validation_P_recall) + ', ' + str(validation_N_recall) + ', ' + ']')
            print(str(epoch + 1) + 'epoch ' + str(total_step) + 'test region [small, middle, large]: ['
                  + str(test_region_acc[0]) + ', ' + str(test_region_acc[1]) + ', ' + str(test_region_acc[2]) + ']')
            # ===== 保存 =====
            eva_set = [total_step, train_accuracy, test_accuracy, validation_accuracy,
                       train_P_precision, train_N_precision,
                       train_P_recall, train_N_recall,
                       test_P_precision, test_N_precision,
                       test_P_recall, test_N_recall,
                       validation_P_precision, validation_N_precision,
                       validation_P_recall, validation_N_recall,
                       train_confusion_matrix[0], train_confusion_matrix[1],
                       train_confusion_matrix[2], train_confusion_matrix[3],
                       test_confusion_matrix[0], test_confusion_matrix[1],
                       test_confusion_matrix[2], test_confusion_matrix[3],
                       validation_confusion_matrix[0], validation_confusion_matrix[1],
                       validation_confusion_matrix[2], validation_confusion_matrix[3],
                       train_region_acc[0], train_region_acc[1], train_region_acc[2],
                       test_region_acc[0], test_region_acc[1], test_region_acc[2],
                       validation_region_acc[0], validation_region_acc[1], validation_region_acc[2],
                       train_gen_loss, test_gen_loss, validation_gen_loss,]
            with open(csv_file, 'w') as f:
                writer_total = csv.writer(f, lineterminator='\n')
                writer_total.writerow(eva_set)


    # ==============================================================================================================
    # ========== check_accuracy関数 ================================================================================
    # ==============================================================================================================
    # 再現率対応
    def check_accuracy(self, ds_cls, device_list, task_name='check_accuracy'):
        total_accuracy = 0
        total_region_acc = np.array([0., 0., 0., ])
        total_confusion_matrix = np.array([0., 0., 0., 0., ])  # TP, FP, FN, TN
        total_loss = 0
        ds_iter = ds_cls.get_inited_iter()
        pbar = tqdm(total=ds_cls.get_remain_data(), desc=task_name, leave=False)  # プロセスバーの設定
        while ds_cls.get_remain_data() is not 0:
            if ds_cls.get_remain_data() >= (ds_cls.get_batch_size() * len(device_list)):
                # マルチデバイス
                # accuracy, confusion_matrix, region_acc = self.multi_check_step(ds_iter=ds_iter, device_list=device_list, data_n=ds_cls.get_next_data_n())
                accuracy, confusion_matrix, region_acc, gen_loss = self.multi_check_step(ds_iter=ds_iter,
                                                                                         device_list=device_list,
                                                                                         data_n=ds_cls.get_next_data_n())
                ds_cls.data_used_apply(use_count=len(device_list))
                progress_step = len(device_list) * ds_cls.get_batch_size()
            else:
                # シングルデバイス
                accuracy, confusion_matrix, region_acc, gen_loss = self.check_step(ds_iter=ds_iter,
                                                                                   gpu_index=device_list[0],
                                                                                   data_n=ds_cls.get_next_data_n())
                ds_cls.data_used_apply(use_count=1)
                progress_step = ds_cls.get_next_data_n()
            total_accuracy += accuracy
            total_confusion_matrix += np.array(confusion_matrix)
            total_region_acc += np.array(region_acc)
            total_loss += gen_loss
            pbar.update(progress_step)  # プロセスバーを進行
        # print(confusion_matrix)cy / ds_cls.get_total_data()
        ave_accuracy = total_accuracy / ds_cls.get_total_data()
        total_region_acc = total_region_acc / ds_cls.get_total_data()
        ave_loss = total_loss / ds_cls.get_total_data()
        pbar.close()  # プロセスバーの終了
        return ave_accuracy.numpy(), total_confusion_matrix, total_region_acc, ave_loss.numpy()

    def multi_check_step(self, ds_iter, device_list, data_n):
        accuracy_list = []
        small_acc_list = []
        middle_acc_list = []
        large_acc_list = []
        TP_list = []
        TN_list = []
        FN_list = []
        FP_list = []
        for gpu_index in device_list:
            with tf.device('/gpu:%d' % gpu_index):  # gpu単位の処理
                input_image, target, weight = next(ds_iter)
                target = nfunc.target_cut_padding(target=target, padding=self.net_cls.get_padding())
                # Generatorによる画像生成
                generator = self.net_cls.get_generator()
                gen_output = generator(input_image, training=False)
                gen_loss = self.net_cls.generator_loss(gen_output=gen_output, target=target, weight=weight)
                acc, confusion_matrix, region_acc_list = nfunc.evaluate(net_cls=self.net_cls, out=gen_output,
                                                                        ans=target,
                                                                        region=[self.small_region, self.middle_region,
                                                                                self.large_region],
                                                                        batch_data_n=int(data_n / len(device_list)), )
                # confusion_matrix [TP, FP, FN, TN]
                data_n = tf.cast(data_n, tf.float32)
                accuracy_list.append(acc * data_n)
                small_acc_list.append(region_acc_list[0] * data_n)
                middle_acc_list.append(region_acc_list[1] * data_n)
                large_acc_list.append(region_acc_list[2] * data_n)
                TP_list.append(confusion_matrix[0])
                FP_list.append(confusion_matrix[1])
                FN_list.append(confusion_matrix[2])
                TN_list.append(confusion_matrix[3])
        return sum(accuracy_list), [sum(TP_list), sum(FP_list), sum(FN_list), sum(TN_list)], [sum(small_acc_list),
                                                                                              sum(middle_acc_list), sum(
                large_acc_list)], gen_loss

    # self.small_region = 148
    # self.middle_region = 210
    # self.large_region = 256

    def check_step(self, ds_iter, gpu_index, data_n):
        accuracy_list = []
        with tf.device('/gpu:%d' % gpu_index):  # gpu単位の処理
            input_image, target, weight = next(ds_iter)
            target = nfunc.target_cut_padding(target=target, padding=self.net_cls.get_padding())
            # Generatorによる画像生成
            generator = self.net_cls.get_generator()
            gen_output = generator(input_image, training=False)
            gen_loss = self.net_cls.generator_loss(gen_output=gen_output, target=target, weight=weight)
            acc, confusion_matrix, region_acc_list = nfunc.evaluate(net_cls=self.net_cls, out=gen_output, ans=target,
                                                                    region=[self.small_region, self.middle_region,
                                                                            self.large_region],
                                                                    batch_data_n=int(data_n), )
            # confusion_matrix [TP, FP, FN, TN]
            data_n = tf.cast(data_n, tf.float32)
            acc = (acc * data_n)
            small_acc = (region_acc_list[0] * data_n)
            middle_acc = (region_acc_list[1] * data_n)
            large_region = (region_acc_list[2] * data_n)
        return acc, confusion_matrix, [small_acc, middle_acc, large_region], gen_loss




