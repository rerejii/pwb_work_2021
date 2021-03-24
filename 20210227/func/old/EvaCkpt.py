# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import csv
import time as timefunc
import sys
import os
import pandas as pd
import cv2
from tqdm import tqdm
import subprocess
import math
import gc

work_dirname = (os.path.dirname(os.path.abspath(__file__)))
import ShareNetFunc as nfunc
from ImageGenerator import ImageGenerator

class EvaCkpt:
    def __init__(self, net_cls, path_cls, exe_file,
                 model_hw=[256,256,], fin_activate='sigmoid',
                 standardization_csv_path=None, 
                 standardization=False):
        self.net_cls = net_cls
        self.path_cls = path_cls
        self.exe_file = exe_file
        self.model_h = model_hw[0]
        self.model_w = model_hw[1]
        self.fin_activate = fin_activate
        self.standardization_csv_path = standardization_csv_path
        self.standardization = standardization
        self.csv_header = ['step',
                           'short', 
                           'break+break_same', 
                           'total',
                           'division', 
                           'short_area',
                           'break_area', 
                           'break_same_area']

    def run(self, img_path_set, ans_path_set):
        # imagenameの取得
        img_name_set = []
        for img_index in range(len(img_path_set)):
            s = img_path_set[img_index].find('L00')
            img_n = (img_path_set[img_index][s: s+5])
            img_name_set.append(img_n)

        # workfolderの作成
        work_path = self.path_cls.get_outroot_path() + '/workfolder'
        os.makedirs(work_path, exist_ok=True)

        # csvファイルの作成
        csv_file_names = []
        for img_index in range(len(img_path_set)):
            img_path = img_path_set[img_index]
            img_name = img_name_set[img_index]
            csv_file = self.path_cls.make_csv_path('eva_ckpt_'+img_name+'.csv')
            csv_file_names.append(csv_file)
            if not os.path.isfile(csv_file):
                with open(csv_file, 'w') as f:
                    writer_total = csv.writer(f, lineterminator='\n')
                    writer_total.writerow(self.csv_header)

        # checkpoint の検索
        ckpt_path_set = self.path_cls.search_checkpoint_path()

        # checkpoint の変更
        for ckpt_index in range(len(ckpt_path_set)):
            print('========== now ckpt: ' +str(ckpt_index+1)+ ' / '+ str(len(ckpt_path_set)) + ' ==========')
            ckpt_path = ckpt_path_set[ckpt_index]
            self.net_cls.ckpt_restore(ckpt_path)
            step = self.net_cls.get_step()

            # 過去のmodel削除
            model_path = os.path.join(work_path, 'ckpt_model.h5')
            if os.path.exists( model_path ):
                os.remove( model_path )

            # modelの生成
            model_path = os.path.join(work_path, 'ckpt_model.h5')
            self.net_cls.get_generator().save(model_path)

            # 処理する画像リストの生成
            param_csv_path = work_path + '/param.csv'
            img_csv_path =  work_path + '/img.csv'

            if self.standardization_csv_path is None:
                s_csv_path = 'None'

            with open(param_csv_path, 'w') as f:
                writer = csv.writer(f, lineterminator='\n')
                writer.writerow(['model_h', 'model_w', 'padding', 'model_path', 'fin_activate',
                                 'standardization_csv_path', 'standardization', 'work_dirname', 'work_path',])
                writer.writerow([self.model_h, self.model_w, self.net_cls.get_padding(),
                                 model_path, self.fin_activate,
                                 s_csv_path, self.standardization,
                                 work_dirname, work_path,])

            flg = False
            with open(img_csv_path, 'w') as f:
                writer = csv.writer(f, lineterminator='\n')
                writer.writerow(['img_path', 'ans_path'])

                for img_index in range(len(img_path_set)):
                    csv_file = csv_file_names[img_index]
                    df = pd.read_csv(csv_file)
                    if step in df['step'].values:
                        continue
                    flg = True
                    writer.writerow([
                        img_path_set[img_index],
                        ans_path_set[img_index], 
                        ])
            # 実行
            # if flg:
            subprocess.call('python ' + work_dirname + '/eva.py ' + param_csv_path + ' ' + img_csv_path)

