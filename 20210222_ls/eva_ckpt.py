# -*- coding: utf-8 -*-
# main.py GPU番号 EPOCH数 DATASET_PATH OUT_PATH

from __future__ import absolute_import, division, print_function, unicode_literals

import os
import numpy as np
import pandas as pd
import sys
import random
from tqdm import tqdm
from importlib import import_module

# ========== import設定 ==========
work_dirname = (os.path.dirname(os.path.abspath(__file__)))
sys.path.append(work_dirname + '/' + 'net_cls/')
sys.path.append(work_dirname + '/' + 'func/')

# ========== GPU設定 ==========
args = sys.argv
DEVICE_NUMBER_STR = args[3]  # 使用するGPU設定
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
DATASET = args[5]
# 出力ネーム
MAINNAME = 'unet_weight-d5-1_fin-none_logit-true_thr-d0'
# importする net cls
IMPORT_NET_CLS = 'unet_weight-d5-1_fin-none_logit-true_thr-d0'
# sigmoid tanh ...
FIN_ACTIVATE = 'None'
#  '20200223'
OUT_ROOT_FOLDER_NAME = '20201218'
#  '20200223'
EPOCH_OUT_ROOT_FOLDER_NAME = '20201218'
#  DR*
TFRECORD_FOLDER_NAME = 'DR-256-morphology-weight-valid0-more'
#  Img*
IMAGE_ROOT_PATH_NAME = 'Img-256-valid0-more'

# ========== 定数2(たまに修正するシリーズ) ==========
GENERATOR_TEST_PATH = '/home/hayakawa/work/myTensor/dataset2/17H-0863-1_L0002-new/L0002.png'
# MAINNAME+'-'+DATASET+'.h5'
CYCLE_LENGHT = 4  # データ読み込み処理オーバーラップ数
SHUF_LEARN_BATCH_SIZE = 20
LEARN_BATCH_SIZE = 50
VALIDATION_BATCH_SIZE = 50
TEST_BATCH_SIZE = 50
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256
SUFFLE_BUFFER_SIZE = 100000
OUTPUT_CHANNELS = 1
CHECK_STEP = 21888
END_EPOCH = int(args[4])

# ========== 調整 ==========
DATASET_PATH = args[1]
OUT_PATH = args[2]
OUT_ROOT_FOLDER = OUT_PATH + '/' + OUT_ROOT_FOLDER_NAME + '/' + MAINNAME+'/'+MAINNAME+ '-' + DATASET
GENERATOR_OUTPATH = OUT_ROOT_FOLDER + '/out.png'
EPOCH_OUT_ROOT_FOLDER = OUT_PATH+ '/' + EPOCH_OUT_ROOT_FOLDER_NAME + '_epoch' + '/' + MAINNAME +\
                        '/'+MAINNAME+ '-' + DATASET if EPOCH_OUT_ROOT_FOLDER_NAME is not '' else None
TFRECORD_FOLDER = DATASET_PATH + '/' + TFRECORD_FOLDER_NAME + '/' + DATASET
IMAGE_ROOT_PATH = DATASET_PATH + '/' + IMAGE_ROOT_PATH_NAME
STANDARDIZATION_CSV_PATH = None
# standardization_csv_path
# ========== 確認用画像集 ==========
CHECK_SAMPLE_PATH = [

]
CHECK_ANSWER_PATH = [

]
CHECK_IMG_TAG = [

] # 'test', 'valid', 'learn'
check_img_path = []
for i in range(len(CHECK_SAMPLE_PATH)):
    check_img_path.append([IMAGE_ROOT_PATH+'/'+CHECK_SAMPLE_PATH[i],
                           IMAGE_ROOT_PATH+'/'+CHECK_ANSWER_PATH[i],
                           CHECK_IMG_TAG[i],
                           ],)

# ========== My class inport ==========
# from EncodeManager import NetManager  # 呼び出すネットによって変える
NetManager = import_module(IMPORT_NET_CLS).NetManager
from DataManager import DataManager
from PathManager import PathManager
from ImageGenerator import ImageGenerator
from FitManager import FitManager
import ShareNetFunc as nfunc
import PlotFunc
from EvaCkpt import EvaCkpt

# ========== NetManager呼び出し ==========
net_cls = NetManager()

# ========== PathManager呼び出し ==========
path_cls = PathManager(tfrecord_folder=TFRECORD_FOLDER,
                       output_rootfolder=OUT_ROOT_FOLDER,
                       epoch_output_rootfolder=EPOCH_OUT_ROOT_FOLDER)
path_cls.all_makedirs()  # 結果保存フォルダ生成

# ========== DataSet呼び出し ==========
# プロパティデータ読み込み
df = pd.read_csv(path_cls.get_property_path())
shuf_train_ds_cls = DataManager(tfrecord_path=path_cls.get_train_ds_path(),
                                img_root=IMAGE_ROOT_PATH,
                                batch_size=SHUF_LEARN_BATCH_SIZE,
                                net_cls=net_cls,
                                data_n=df.at[0, 'total_learn_data'],
                                suffle_buffer=SUFFLE_BUFFER_SIZE,
                                standardization_csv_path=STANDARDIZATION_CSV_PATH,
                                )
train_ds_cls = DataManager(tfrecord_path=path_cls.get_train_ds_path(),
                           img_root=IMAGE_ROOT_PATH,
                           batch_size=LEARN_BATCH_SIZE,
                           net_cls=net_cls,
                           data_n=df.at[0, 'total_learn_data'],
                           standardization_csv_path=STANDARDIZATION_CSV_PATH,
                           )
test_ds_cls = DataManager(tfrecord_path=path_cls.get_test_ds_path(),
                          img_root=IMAGE_ROOT_PATH,
                          batch_size=TEST_BATCH_SIZE,
                          net_cls=net_cls,
                          data_n=df.at[0, 'total_test_data'],
                          standardization_csv_path=STANDARDIZATION_CSV_PATH,
                          )
valid_ds_cls = DataManager(tfrecord_path=path_cls.get_validation_ds_path(),
                          img_root=IMAGE_ROOT_PATH,
                          batch_size=VALIDATION_BATCH_SIZE,
                          net_cls=net_cls,
                          data_n=df.at[0, 'total_valid_data'],
                          standardization_csv_path=STANDARDIZATION_CSV_PATH,
                          )

# ========== plot ==========
PlotFunc.accuracy_plot(
    path_cls=path_cls,
    title=OUT_ROOT_FOLDER + '/[TEST]_' + os.path.basename(OUT_ROOT_FOLDER),
    label=DATASET,
    item='test_accuracy',
)
PlotFunc.accuracy_plot(
    path_cls=path_cls,
    title=OUT_ROOT_FOLDER + '/[TRAIN]_' + os.path.basename(OUT_ROOT_FOLDER),
    label=DATASET,
    item='train_accuracy',
)
PlotFunc.accuracy_plot(
    path_cls=path_cls,
    title=OUT_ROOT_FOLDER + '/[VALIDATION]_' + os.path.basename(OUT_ROOT_FOLDER),
    label=DATASET,
    item='validation_accuracy',
)
PlotFunc.multi_accuracy_plot(
    path_cls=path_cls,
    title=OUT_ROOT_FOLDER + '/[ALL]_' + os.path.basename(OUT_ROOT_FOLDER),
    label=DATASET,
    items=['train_accuracy', 'test_accuracy', 'validation_accuracy'],
)

# ========== 生成速度計測 ==========
if DATASET == 'A':
    GENERATOR_IMAGE = ['img/17H-0863-1_L0006-new/L0006.png', 'img/17H-0863-1_L0011/L0011.png', 'img/17H-0863-1_L0017/L0017.png']
    ANSWER_IMAGE = ['img/17H-0863-1_L0006-new/L0006_bin.png', 'img/17H-0863-1_L0011/L0011_bin.png', 'img/17H-0863-1_L0017/L0017_bin.png']
    GENERATOR_IMAGE_NAME = ['generator_L0006', 'generator_L0011', 'generator_L0017']
elif DATASET == 'B':
    GENERATOR_IMAGE = ['img/17H-0863-1_L0003/L0003.png', 'img/17H-0863-1_L0005/L0005.png', 'img/17H-0863-1_L0009/L0009.png']
    ANSWER_IMAGE = ['img/17H-0863-1_L0003/L0003_bin.png', 'img/17H-0863-1_L0005/L0005_bin.png', 'img/17H-0863-1_L0009/L0009_bin.png']
    GENERATOR_IMAGE_NAME = ['generator_L0003', 'generator_L0005', 'generator_L0009']
elif DATASET == 'C':
    GENERATOR_IMAGE = ['img/17H-0863-1_L0002-old/L0002.png', 'img/17H-0863-1_L0008/L0008.png', 'img/17H-0863-1_L0013/L0013.png']
    ANSWER_IMAGE = ['img/17H-0863-1_L0002-old/L0002_bin.png', 'img/17H-0863-1_L0008/L0008_bin.png', 'img/17H-0863-1_L0013/L0013_bin.png']
    GENERATOR_IMAGE_NAME = ['generator_L0002', 'generator_L0008', 'generator_L0013']
elif DATASET == 'D':
    GENERATOR_IMAGE = ['img/17H-0863-1_L0007/L0007.png', 'img/17H-0863-1_L0012/L0012.png', 'img/17H-0863-1_L0014/L0014.png']
    ANSWER_IMAGE = ['img/17H-0863-1_L0007/L0007_bin.png', 'img/17H-0863-1_L0012/L0012_bin.png', 'img/17H-0863-1_L0014/L0014_bin.png']
    GENERATOR_IMAGE_NAME = ['generator_L0007', 'generator_L0012', 'generator_L0014']
elif DATASET == 'E':
    GENERATOR_IMAGE = ['img/17H-0863-1_L0004/L0004.png', 'img/17H-0863-1_L0015/L0015.png', 'img/17H-0863-1_L0016/L0016.png']
    ANSWER_IMAGE = ['img/17H-0863-1_L0004/L0004_bin.png', 'img/17H-0863-1_L0015/L0015_bin.png', 'img/17H-0863-1_L0016/L0016_bin.png']
    GENERATOR_IMAGE_NAME = ['generator_L0004', 'generator_L0015', 'generator_L0016']
elif DATASET == 'F':
    GENERATOR_IMAGE = ['img/17H-0863-1_L0001/L0001.png', 'img/17H-0863-1_L0010/L0010.png', 'img/17H-0863-1_L0018/L0018.png']
    ANSWER_IMAGE = ['img/17H-0863-1_L0001/L0001_bin.png', 'img/17H-0863-1_L0010/L0010_bin.png', 'img/17H-0863-1_L0018/L0018_bin.png']
    GENERATOR_IMAGE_NAME = ['generator_L0001', 'generator_L0010', 'generator_L0018']

GENERATOR_IMAGE = [DATASET_PATH + '/' + path for path in GENERATOR_IMAGE]
ANSWER_IMAGE = [DATASET_PATH + '/' + path for path in ANSWER_IMAGE]

evackpt_cls = EvaCkpt(
    net_cls=net_cls, 
    path_cls=path_cls, 
    exe_file=work_dirname+'/PatternEva_1130.exe',
    model_hw=[IMAGE_HEIGHT, IMAGE_WIDTH], 
    fin_activate=FIN_ACTIVATE,
)

evackpt_cls.run(img_path_set=GENERATOR_IMAGE, ans_path_set=ANSWER_IMAGE)