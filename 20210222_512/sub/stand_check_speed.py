# -*- coding: utf-8 -*-
# main.py GPU番号 EPOCH数 DATASET_PATH OUT_PATH

from __future__ import absolute_import, division, print_function, unicode_literals

import os
import numpy as np
import pandas as pd
import sys
import random
import glob
import csv
from tqdm import tqdm
from importlib import import_module
from natsort import natsorted

# ========== import設定 ==========
work_dirname = (os.path.dirname(os.path.abspath(__file__)))
sys.path.append(work_dirname + '/' + 'net_cls/')
sys.path.append(work_dirname + '/' + 'func/')

# ========== GPU設定 ==========
args = sys.argv
DEVICE_NUMBER_STR = args[4]  # 使用するGPU設定
# device番号のlistを生成(複数GPU学習時の割り当て調整に使用)
DEVICE_STR_LIST = DEVICE_NUMBER_STR.split(',')
# DEVICE_LIST = [int(n) for n in DEVICE_STR_LIST]
DEVICE_LIST = [int(n) for n in range(len(DEVICE_STR_LIST))]
# 環境変数の調整
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = DEVICE_NUMBER_STR

# ========== TensorFlow呼び出し ==========
# 環境変数の変更を適用したのちに呼び出す
import tensorflow as tf

# ========== 乱数シード指定 ==========
seed = 1
os.environ['PYTHONHASHSEED'] = str(seed)  # https://docs.python.org/ja/3.6/using/cmdline.html#envvar-PYTHONHASHSEED
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

# ========== 定数1(ほぼ毎回変更するシリーズ) ==========
# ネットワークのpaddingサイズも調整したか？
# A B C D ...
# DATASET = args[6]
# 出力ネーム
MAINNAME = args[1]
# importする net cls
IMPORT_NET_CLS = args[2]
# sigmoid tanh ...
FIN_ACTIVATE = 'sigmoid'
#  '20200223'
OUT_ROOT_FOLDER_NAME = '20200927'
#  '20200223'
EPOCH_OUT_ROOT_FOLDER_NAME = '20200927'
#  DR*
TFRECORD_FOLDER_NAME = 'DR-256-valid0-more'
#  Img*
IMAGE_ROOT_PATH_NAME = 'Img-256-valid0-more'

# ========== 定数2(たまに修正するシリーズ) ==========
GENERATOR_TEST_PATH = '/home/hayakawa/work/myTensor/dataset2/17H-0863-1_L0002-new/L0002.png'
# MAINNAME+'-'+DATASET+'.h5'
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256

# ========== 調整 ==========
DATASET_PATH = args[3]
OUT_PATH = args[4]
CSV_NAME = args[5]
REPEAT = int(args[6])

# ========== My class inport ==========
# from EncodeManager import NetManager  # 呼び出すネットによって変える
NetManager = import_module(IMPORT_NET_CLS).NetManager
from DataManager import DataManager
from PathManager import PathManager
from ImageGenerator import ImageGenerator
from FitManager import FitManager
import ShareNetFunc as nfunc
import PlotFunc

for DATASET in ['A', 'B', 'C', 'D', 'E', 'F']:
    STANDARDIZATION_CSV_PATH = DATASET_PATH + '/std/std-' + DATASET + '.csv'
    OUT_ROOT_FOLDER = OUT_PATH + '/' + OUT_ROOT_FOLDER_NAME + '/' + MAINNAME+'/'+MAINNAME+ '-' + DATASET
    GENERATOR_OUTPATH = OUT_ROOT_FOLDER + '/out.png'
    EPOCH_OUT_ROOT_FOLDER = OUT_PATH+ '/' + EPOCH_OUT_ROOT_FOLDER_NAME + '_epoch' + '/' + MAINNAME +\
                            '/'+MAINNAME+ '-' + DATASET 
    TFRECORD_FOLDER = DATASET_PATH + '/' + TFRECORD_FOLDER_NAME + '/' + DATASET
    IMAGE_ROOT_PATH = DATASET_PATH + '/' + IMAGE_ROOT_PATH_NAME

    # ========== NetManager呼び出し ==========
    net_cls = NetManager()

    # ========== PathManager呼び出し ==========
    path_cls = PathManager(tfrecord_folder=TFRECORD_FOLDER,
                        output_rootfolder=OUT_ROOT_FOLDER,
                        epoch_output_rootfolder=EPOCH_OUT_ROOT_FOLDER)
    path_cls.all_makedirs()  # 結果保存フォルダ生成

    # ========== save model ==========
    modelpath = path_cls.make_model_path(MAINNAME + '-' + DATASET + '.h5')
    # nfunc.save_best_generator_model(net_cls=net_cls, path_cls=path_cls, path=modelpath)

    # TOTAL_CSV = path_cls.make_csv_path('../../Total_time_'+CSV_NAME+'_cpu'+'.csv')
    # if os.path.exists(TOTAL_CSV):
    #     sys.exit()


    # ========== 生成速度計測 ==========
    if DATASET == 'A':
        GENERATOR_IMAGE = ['img/17H-0863-1_L0006-new/L0006.png', 'img/17H-0863-1_L0011/L0011.png', 'img/17H-0863-1_L0017/L0017.png']
        GENERATOR_IMAGE_NAME = ['generator_L0006', 'generator_L0011', 'generator_L0017']
    elif DATASET == 'B':
        GENERATOR_IMAGE = ['img/17H-0863-1_L0003/L0003.png', 'img/17H-0863-1_L0005/L0005.png', 'img/17H-0863-1_L0009/L0009.png']
        GENERATOR_IMAGE_NAME = ['generator_L0003', 'generator_L0005', 'generator_L0009']
    elif DATASET == 'C':
        GENERATOR_IMAGE = ['img/17H-0863-1_L0002-old/L0002.png', 'img/17H-0863-1_L0008/L0008.png', 'img/17H-0863-1_L0013/L0013.png']
        GENERATOR_IMAGE_NAME = ['generator_L0002', 'generator_L0008', 'generator_L0013']
    elif DATASET == 'D':
        GENERATOR_IMAGE = ['img/17H-0863-1_L0007/L0007.png', 'img/17H-0863-1_L0012/L0012.png', 'img/17H-0863-1_L0014/L0014.png']
        GENERATOR_IMAGE_NAME = ['generator_L0007', 'generator_L0012', 'generator_L0014']
    elif DATASET == 'E':
        GENERATOR_IMAGE = ['img/17H-0863-1_L0004/L0004.png', 'img/17H-0863-1_L0015/L0015.png', 'img/17H-0863-1_L0016/L0016.png']
        GENERATOR_IMAGE_NAME = ['generator_L0004', 'generator_L0015', 'generator_L0016']
    elif DATASET == 'F':
        GENERATOR_IMAGE = ['img/17H-0863-1_L0001/L0001.png', 'img/17H-0863-1_L0010/L0010.png', 'img/17H-0863-1_L0018/L0018.png']
        GENERATOR_IMAGE_NAME = ['generator_L0001', 'generator_L0010', 'generator_L0018']

    gen_cls = ImageGenerator(Generator_model=modelpath,
                        model_h=IMAGE_HEIGHT,
                        model_w=IMAGE_WIDTH,
                        fin_activate=FIN_ACTIVATE,
                        padding=net_cls.get_padding(),
                        use_gpu=False,
                        standardization_csv_path=STANDARDIZATION_CSV_PATH,)

    for i in range(len(GENERATOR_IMAGE)):
        time_out_path = path_cls.make_csv_path('Time_'+CSV_NAME+'_'+GENERATOR_IMAGE_NAME[i]+'_cpu'+'.csv')
        ave_time_out_path = path_cls.make_csv_path('Avetime_'+CSV_NAME+'_'+GENERATOR_IMAGE_NAME[i]+'_cpu'+'.csv')

        if os.path.exists(time_out_path):
            os.remove(time_out_path)
        if os.path.exists(ave_time_out_path):
            os.remove(ave_time_out_path)

        for j in range(REPEAT):
            print('===== [' + DATASET + '] ' + GENERATOR_IMAGE_NAME[i] + ' REPEAT:' +  str(j) + '/' + str(REPEAT) + ' =====')
            gen_cls.run(img_path=DATASET_PATH + '/' + GENERATOR_IMAGE[i],
                        out_path=OUT_ROOT_FOLDER + '/' + GENERATOR_IMAGE_NAME[i] + '_cpu.png',
                        time_out_path=time_out_path,
                        ave_time_out_path=ave_time_out_path,
                        )

ave_list = []
csv_files = glob.glob(OUT_PATH + '/' + OUT_ROOT_FOLDER_NAME + '/' + MAINNAME + '/*/CsvDatas/' +'Avetime_'+CSV_NAME+'*_cpu'+'.csv')
csv_files = natsorted(csv_files)
# print(OUT_PATH + '/' + OUT_ROOT_FOLDER_NAME + '/' + MAINNAME + '/*/' +'Avetime_'+CSV_NAME+'*_cpu'+'.csv')

with open( path_cls.make_csv_path('../../Total_time_'+CSV_NAME+'_cpu'+'.csv'), 'w') as f:
    writer = csv.writer(f, lineterminator='\n')
    csv_header = ['image_name', 'gen_ave_time(str)']
    writer.writerow(csv_header)
    for i in range(len(csv_files)):
        # f_name = os.path.basename(csv_files).split('.', 1)[0]
        s = csv_files[i].find('L00')
        df = pd.read_csv(csv_files[i])
        ave_val = df['gen_ave_time'].values[-1]
        ave_str = df['gen_ave_time(str)'].values[-1]
        writer.writerow( [csv_files[s:s+5], ave_str] )
        ave_list.append(ave_val)
    total_average = np.average(ave_list)
    writer.writerow(['Total', gen_cls.get_str_time(total_average)])

