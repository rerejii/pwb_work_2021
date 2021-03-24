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
from natsort import natsorted

work_dirname = (os.path.dirname(os.path.abspath(__file__)))
import ShareNetFunc as nfunc
from ImageGenerator import ImageGenerator
from Notification import notice, slack_notice, line_notice

class EvaCkpt:
    def __init__(self, net_cls, path_cls, exe_file,
                 model_hw=[256,256,], fin_activate='sigmoid',
                 standardization_csv_path=None, 
                 standardization=False,
                 limit_label=30000,
                 check_step=False,
                 epoch_step=87552):
        self.net_cls = net_cls
        self.path_cls = path_cls
        self.exe_file = exe_file
        self.model_h = model_hw[0]
        self.model_w = model_hw[1]
        self.fin_activate = fin_activate
        self.standardization_csv_path = standardization_csv_path
        self.standardization = standardization
        self.limit_label = limit_label
        self.check_step = check_step
        self.epoch_step = epoch_step
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
                    # os.path.splitext(os.path.basename(img_path))[0]
            label_csv = os.path.splitext(csv_file)[0] + '_label.csv'
            if not os.path.isfile(label_csv):
                with open(label_csv, 'w') as f:
                    writer = csv.writer(f, lineterminator='\n')
                    writer.writerow(['step', 'label'])

        # checkpoint の検索
        ckpt_path_set = self.path_cls.search_checkpoint_path()
        ckpt_path_set = natsorted(ckpt_path_set)
        ckpt_path_set.reverse()

        gen_cls = None
        for i in range(len(ckpt_path_set)):
            ckpt_path = ckpt_path_set[i]
            start = ckpt_path.find('ckpt-step') + 9
            end = ckpt_path.find('-', start)
            ckpt_step = int(ckpt_path[start: end])
            # ckpt_step = 1006852

            for img_index in range(len(img_path_set)):
                csv_file = csv_file_names[img_index]
                df = pd.read_csv(csv_file)
                # if self.check_step and ckpt_step % self.epoch_step != 0:
                #     continue

                if ckpt_step in df['step'].values:
                    continue

                if gen_cls is None:
                    print('==== start ckpt ' + str(ckpt_step) + '====')
                    # 過去のmodel削除
                    modelpath = os.path.join(work_path, 'ckpt_model.h5')
                    if os.path.exists( modelpath ):
                        os.remove( modelpath )

                    # ckptの呼び出し
                    self.net_cls.ckpt_restore(ckpt_path)

                    # modelの生成
                    modelpath = os.path.join(work_path, 'ckpt_model.h5')
                    self.net_cls.get_generator().save(modelpath)

                    # modelの更新
                    gen_cls = ImageGenerator(model_h=self.model_h,
                                            model_w=self.model_w,
                                            padding=self.net_cls.get_padding(),
                                            Generator_model=modelpath,
                                            fin_activate=self.fin_activate,
                                            standardization_csv_path=self.standardization_csv_path,
                                            standardization=self.standardization,)

                print('---- task image: ' +str(img_index+1)+ ' / '+ str(len(img_path_set)) + ' ----')
                self.task(ckpt_step, img_index, csv_file_names, img_path_set, ans_path_set, img_name_set, work_path, gen_cls)
            
            if gen_cls is not None:
                print('==== end ckpt ' + str(ckpt_step) + '====')
                print('==== remaining ' + str(len(ckpt_path_set) - (i+1)) + '====')
                return

        print('!!!! finish !!!!')
        notice('EvaCkpt Finish!')
        return

    def task(self, step, img_index, csv_file_names, img_path_set, ans_path_set, img_name_set, work_path, gen_cls):
        # ==== 画像の生成 ====
        img_path = img_path_set[img_index]
        ans_path = ans_path_set[img_index]
        img_name = img_name_set[img_index]
        csv_file = csv_file_names[img_index]
        out_path = os.path.join(work_path, img_name + '_out.png')
        gen_cls.run(img_path=img_path, out_path=out_path)
        # sys.exit()
        # ==== 画像を4分割 ====

        # ==== ラベルカウント ====
        gray_src = cv2.imread(out_path, 0)
        out_label, _ = cv2.connectedComponents(gray_src)
        label_csv = os.path.splitext(csv_file)[0] + '_label.csv'
        with open(label_csv, 'a') as f:
            writer = csv.writer(f, lineterminator='\n')
            writer.writerow([step, out_label])
        print('==== out label is: ' + str(out_label) +' ====')

        if out_label <= self.limit_label:
            # ==== 短絡・断線の確認 ====
            # Evaluate.txtの削除
            evaluate_path = os.path.join(work_path, 'Evaluate.txt')
            if os.path.exists(evaluate_path):
                os.remove(evaluate_path)
            # PatternEva.exeを実行
            # mark = ['A', 'B', 'C', 'D',]
            # for m in range(len(mark)):
            subprocess.call(self.exe_file + ' ' + ans_path + ' ' + out_path)
            eveluate_result = nfunc.check_eveluate(path=work_path, file_count=1)
        else:
            print('Out label is over the limit!')
            eveluate_result = None
        # csvファイルに追記
        with open(csv_file, 'a') as f:
            writer = csv.writer(f, lineterminator='\n')
            if eveluate_result is not None:
                writer.writerow([
                    step,
                    eveluate_result[2],
                    eveluate_result[4] + eveluate_result[6],
                    eveluate_result[2] + eveluate_result[4] + eveluate_result[6],
                    eveluate_result[0],
                    eveluate_result[2],
                    eveluate_result[4],
                    eveluate_result[6],
                ])
            else:
                writer.writerow([
                    step,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    ])

    # def task(self, step, img_index, csv_file_names, img_path_set, ans_path_set, img_name_set, work_path, gen_cls):
    #     # ==== 画像の生成 ====
    #     img_path = img_path_set[img_index]
    #     ans_path = ans_path_set[img_index]
    #     img_name = img_name_set[img_index]
    #     out_path = os.path.join(work_path, img_name + '_out.png')
    #     gen_cls.run(img_path=img_path, out_path=out_path)
    #     # ==== 画像を4分割 ====
    #     # out_img
    #     out_img = cv2.imread(out_path, cv2.IMREAD_GRAYSCALE)
    #     height, width = out_img.shape
    #     half_h = math.floor(height / 2)
    #     half_w = math.floor(width / 2)
    #     cut_out_A = out_img[0: half_h, 0: half_w]
    #     cut_out_B = out_img[0: half_h, half_w:]
    #     cut_out_C = out_img[half_h: , 0: half_w]
    #     cut_out_D = out_img[half_h: , half_w:]
    #     cut_out_paths = [
    #         os.path.join(work_path, img_name + '_outA.png'),
    #         os.path.join(work_path, img_name + '_outB.png'),
    #         os.path.join(work_path, img_name + '_outC.png'),
    #         os.path.join(work_path, img_name + '_outD.png'),
    #     ]
    #     cv2.imwrite(cut_out_paths[0], cut_out_A)
    #     cv2.imwrite(cut_out_paths[1], cut_out_B)
    #     cv2.imwrite(cut_out_paths[2], cut_out_C)
    #     cv2.imwrite(cut_out_paths[3], cut_out_D)
    #     # ans_img
    #     ans_img = cv2.imread(ans_path, cv2.IMREAD_GRAYSCALE)
    #     cut_ans_A = ans_img[0: half_h, 0: half_w]
    #     cut_ans_B = ans_img[0: half_h, half_w:]
    #     cut_ans_C = ans_img[half_h: , 0: half_w]
    #     cut_ans_D = ans_img[half_h: , half_w:]
    #     cut_ans_paths = [
    #         os.path.join(work_path, img_name + '_binA.png'),
    #         os.path.join(work_path, img_name + '_binB.png'),
    #         os.path.join(work_path, img_name + '_binC.png'),
    #         os.path.join(work_path, img_name + '_binD.png'),
    #     ]
    #     cv2.imwrite(cut_ans_paths[0], cut_ans_A)
    #     cv2.imwrite(cut_ans_paths[1], cut_ans_B)
    #     cv2.imwrite(cut_ans_paths[2], cut_ans_C)
    #     cv2.imwrite(cut_ans_paths[3], cut_ans_D)

    #     # ==== 短絡・断線の確認 ====
    #     # Evaluate.txtの削除
    #     evaluate_path = os.path.join(work_path, 'Evaluate.txt')
    #     if os.path.exists( evaluate_path ):
    #         os.remove( evaluate_path )
    #     # PatternEva.exeを実行
    #     mark = ['A', 'B', 'C', 'D',]
    #     for m in range(len(mark)):
    #         subprocess.call(self.exe_file + ' ' + cut_ans_paths[m] + ' ' + cut_out_paths[m])
    #     eveluate_result = nfunc.check_eveluate(path=work_path, file_count=len(mark))
    #     # csvファイルに追記
    #     with open(csv_file, 'a') as f:
    #         writer = csv.writer(f, lineterminator='\n')
    #         if eveluate_result is not None:
    #             writer.writerow([
    #                 step,
    #                 eveluate_result[2],
    #                 eveluate_result[4] + eveluate_result[6],
    #                 eveluate_result[2] + eveluate_result[4] + eveluate_result[6],
    #                 eveluate_result[0],
    #                 eveluate_result[2],
    #                 eveluate_result[4],
    #                 eveluate_result[6],
    #             ])
    #         else:
    #             writer.writerow([
    #                 step,
    #                 'None',
    #                 'None',
    #                 'None',
    #                 'None',
    #                 'None',
    #                 'None',
    #                 'None',
    #                 ])
