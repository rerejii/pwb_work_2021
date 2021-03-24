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

args = sys.argv
param_csv_path = args[1]  # csv
img_csv_path = args[2]
p_df = pd.read_csv(param_csv_path)

# work_dirname = p_df['work_dirname'][0]
# sys.path.append(work_dirname + '/' + 'func/')
import ShareNetFunc as nfunc
from ImageGenerator import ImageGenerator


if not os.path.exists( p_df['standardization_csv_path'][0]):
    s_csv_path = None
else:
    s_csv_path = p_df['standardization_csv_path'][0]

if len(p_df.index) is not 0:
    gen_cls = ImageGenerator(model_h=p_df['model_h'][0],
                            model_w=p_df['model_w'][0],
                            padding=p_df['padding'][0],
                            Generator_model=p_df['model_path'][0],
                            fin_activate=p_df['fin_activate'][0],
                            standardization_csv_path=None,
                            standardization=False,
                            )

work_path = p_df['work_path'][0]
df = pd.read_csv(img_csv_path)
for i in range(len(df.index)):
    img_path = df['img_path'][i]
    ans_path = df['ans_path'][i]
    print('---- task image: ' +str(i+1)+ ' / '+ str(len(df.index)) + ' ----')
    s = img_path.find('L00')
    img_name = img_path[s: s+5]
    out_path = os.path.join(work_path, img_name + '_out.png')
    gen_cls.run(img_path=img_path, out_path=out_path)
    # ==== 画像を4分割 ====
    out_img = cv2.imread(out_path, cv2.IMREAD_GRAYSCALE)
    height, width = out_img.shape
    half_h = math.floor(height / 2)
    half_w = math.floor(width / 2)
    cut_out_A = out_img[0: half_h, 0: half_w]
    cut_out_B = out_img[0: half_h, half_w:]
    cut_out_C = out_img[half_h: , 0: half_w]
    cut_out_D = out_img[half_h: , half_w:]
    cut_out_paths = [
        os.path.join(work_path, img_name + '_outA.png'),
        os.path.join(work_path, img_name + '_outB.png'),
        os.path.join(work_path, img_name + '_outC.png'),
        os.path.join(work_path, img_name + '_outD.png'),
    ]
    cv2.imwrite(cut_out_paths[0], cut_out_A)
    cv2.imwrite(cut_out_paths[1], cut_out_B)
    cv2.imwrite(cut_out_paths[2], cut_out_C)
    cv2.imwrite(cut_out_paths[3], cut_out_D)
    # ans_img
    ans_img = cv2.imread(ans_path, cv2.IMREAD_GRAYSCALE)
    cut_ans_A = ans_img[0: half_h, 0: half_w]
    cut_ans_B = ans_img[0: half_h, half_w:]
    cut_ans_C = ans_img[half_h: , 0: half_w]
    cut_ans_D = ans_img[half_h: , half_w:]
    cut_ans_paths = [
        os.path.join(work_path, img_name + '_binA.png'),
        os.path.join(work_path, img_name + '_binB.png'),
        os.path.join(work_path, img_name + '_binC.png'),
        os.path.join(work_path, img_name + '_binD.png'),
    ]
    cv2.imwrite(cut_ans_paths[0], cut_ans_A)
    cv2.imwrite(cut_ans_paths[1], cut_ans_B)
    cv2.imwrite(cut_ans_paths[2], cut_ans_C)
    cv2.imwrite(cut_ans_paths[3], cut_ans_D)

    # ==== 短絡・断線の確認 ====
    # Evaluate.txtの削除
    evaluate_path = os.path.join(work_path, 'Evaluate.txt')
    if os.path.exists( evaluate_path ):
        os.remove( evaluate_path )
    # PatternEva.exeを実行
    mark = ['A', 'B', 'C', 'D',]
    for m in range(len(mark)):
        subprocess.call(self.exe_file + ' ' + cut_ans_paths[m] + ' ' + cut_out_paths[m])
    eveluate_result = nfunc.check_eveluate(path=work_path, file_count=len(mark))
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
                'None',
                'None',
                'None',
                'None',
                'None',
                'None',
                'None',
                ])
