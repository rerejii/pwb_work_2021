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
import traceback

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

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

# ========== 定数1(ほぼ毎回変更するシリーズ) ==========
# ネットワークのpaddingサイズも調整したか？
# A B C D ...
DATASET = args[5]
# 出力ネーム
MAINNAME = args[7]
# importする net cls
IMPORT_NET_CLS = args[7]

TASK = args[8] # train or eva
# sigmoid tanh ...
FIN_ACTIVATE = 'None'
#  '20200223'
OUT_ROOT_FOLDER_NAME = '20210303'
#  '20200223'
EPOCH_OUT_ROOT_FOLDER_NAME = '20210303'
#  DR*
TFRECORD_FOLDER_NAME = 'DR-256-20210208-valid15-more'
#  Img*
IMAGE_ROOT_PATH_NAME = 'Img-256-valid15-more'
# DS_PADDING
DS_PADDING = 15

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
# CHECK_STEP = int(80256 / 4)
# CHECK_STEP = 171
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
EXE_MANAGE_CSV = OUT_PATH + '/' + OUT_ROOT_FOLDER_NAME + '/' + OUT_ROOT_FOLDER_NAME + '_task.csv'
# standardization_csv_path
# ========== 確認用画像集 ==========
CHECK_SAMPLE_PATH = [
    # '17H-0863-1_L0001/Sample/Sample_X009_Y038.png',
    # '17H-0863-1_L0002-old/Sample/Sample_X002_Y032.png',
    # '17H-0863-1_L0003/Sample/Sample_X004_Y035.png',
    # '17H-0863-1_L0004/Sample/Sample_X008_Y039.png',
    # '17H-0863-1_L0005/Sample/Sample_X009_Y008.png',
    # '17H-0863-1_L0006-new/Sample/Sample_X002_Y015.png',
    # '17H-0863-1_L0007/Sample/Sample_X018_Y006.png',
    # '17H-0863-1_L0008/Sample/Sample_X007_Y026.png',
    # '17H-0863-1_L0009/Sample/Sample_X018_Y067.png',
    # '17H-0863-1_L0010/Sample/Sample_X015_Y064.png',
    # '17H-0863-1_L0011/Sample/Sample_X017_Y052.png',
    # '17H-0863-1_L0012/Sample/Sample_X005_Y033.png',
    # '17H-0863-1_L0013/Sample/Sample_X008_Y074.png',
    # '17H-0863-1_L0014/Sample/Sample_X002_Y069.png',
    # '17H-0863-1_L0015/Sample/Sample_X005_Y004.png',
    # '17H-0863-1_L0016/Sample/Sample_X003_Y036.png',
    # '17H-0863-1_L0017/Sample/Sample_X009_Y071.png',
    # '17H-0863-1_L0018/Sample/Sample_X037_Y022.png',
]
CHECK_ANSWER_PATH = [
    # '17H-0863-1_L0001/Answer/Answer_X009_Y038.png',
    # '17H-0863-1_L0002-old/Answer/Answer_X002_Y032.png',
    # '17H-0863-1_L0003/Answer/Answer_X004_Y035.png',
    # '17H-0863-1_L0004/Answer/Answer_X008_Y039.png',
    # '17H-0863-1_L0005/Answer/Answer_X009_Y008.png',
    # '17H-0863-1_L0006-new/Answer/Answer_X002_Y015.png',
    # '17H-0863-1_L0007/Answer/Answer_X018_Y006.png',
    # '17H-0863-1_L0008/Answer/Answer_X007_Y026.png',
    # '17H-0863-1_L0009/Answer/Answer_X018_Y067.png',
    # '17H-0863-1_L0010/Answer/Answer_X015_Y064.png',
    # '17H-0863-1_L0011/Answer/Answer_X017_Y052.png',
    # '17H-0863-1_L0012/Answer/Answer_X005_Y033.png',
    # '17H-0863-1_L0013/Answer/Answer_X008_Y074.png',
    # '17H-0863-1_L0014/Answer/Answer_X002_Y069.png',
    # '17H-0863-1_L0015/Answer/Answer_X005_Y004.png',
    # '17H-0863-1_L0016/Answer/Answer_X003_Y036.png',
    # '17H-0863-1_L0017/Answer/Answer_X009_Y071.png',
    # '17H-0863-1_L0018/Answer/Answer_X037_Y022.png',
]
CHECK_IMG_TAG = []
# if DATASET == 'A':
#     CHECK_IMG_TAG = ['train', 'train', 'test', 'train', 'test', 'valid', 'train', 'train', 'test', 'train',
#                      'valid', 'train', 'train', 'train', 'train', 'train', 'valid', 'train', ]
# elif DATASET == 'B':
#     CHECK_IMG_TAG = ['train', 'test', 'valid', 'train', 'valid', 'train', 'train', 'test', 'valid', 'train',
#                      'train', 'train', 'test', 'train', 'train', 'train', 'train', 'train', ]
# elif DATASET == 'C':
#     CHECK_IMG_TAG = ['train', 'valid', 'train', 'train', 'train', 'train', 'test', 'valid', 'train', 'train',
#                      'train', 'test', 'valid', 'test', 'train', 'train', 'train', 'train', ]
# elif DATASET == 'D':
#     CHECK_IMG_TAG = ['train', 'train', 'train', 'test', 'train', 'train', 'valid', 'train', 'train', 'train',
#                      'train', 'valid', 'train', 'valid', 'test', 'test', 'train', 'train', ]
# elif DATASET == 'E':
#     CHECK_IMG_TAG = ['test', 'train', 'train', 'valid', 'train', 'train', 'train', 'train', 'train', 'test',
#                      'train', 'train', 'train', 'train', 'valid', 'valid', 'train', 'test', ]
# elif DATASET == 'F':
#     CHECK_IMG_TAG = ['valid', 'train', 'train', 'train', 'train', 'test', 'train', 'train', 'train', 'valid',
#                      'test', 'train', 'train', 'train', 'train', 'train', 'test', 'valid', ]
# CHECK_IMG_TAG = [
#     'train',
#     'test',
#     'valid'
# ] # 'test', 'valid', 'learn'
# CHECK_SAMPLE_PATH = [
#     '17H-0863-1_L0002-old/Sample/Sample_X002_Y032.png',
#     '17H-0863-1_L0003/Sample/Sample_X004_Y035.png',
#     '17H-0863-1_L0006-new/Sample/Sample_X002_Y015.png',
# ]

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
from Notification import notice, slack_notice, line_notice
from EvaCkptAcc import EvaCkptAcc

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
                                padding=DS_PADDING,
                                )
train_ds_cls = DataManager(tfrecord_path=path_cls.get_train_ds_path(),
                           img_root=IMAGE_ROOT_PATH,
                           batch_size=LEARN_BATCH_SIZE,
                           net_cls=net_cls,
                           data_n=df.at[0, 'total_learn_data'],
                           standardization_csv_path=STANDARDIZATION_CSV_PATH,
                           padding=DS_PADDING,
                           )
test_ds_cls = DataManager(tfrecord_path=path_cls.get_test_ds_path(),
                          img_root=IMAGE_ROOT_PATH,
                          batch_size=TEST_BATCH_SIZE,
                          net_cls=net_cls,
                          data_n=df.at[0, 'total_test_data'],
                          standardization_csv_path=STANDARDIZATION_CSV_PATH,
                          padding=DS_PADDING,
                          )
valid_ds_cls = DataManager(tfrecord_path=path_cls.get_validation_ds_path(),
                          img_root=IMAGE_ROOT_PATH,
                          batch_size=VALIDATION_BATCH_SIZE,
                          net_cls=net_cls,
                          data_n=df.at[0, 'total_valid_data'],
                          standardization_csv_path=STANDARDIZATION_CSV_PATH,
                        padding=DS_PADDING,
                          )

# ========== ネットワーク呼び出し ==========
fit_cls = FitManager(net_cls=net_cls,
                     path_cls=path_cls,
                     shuf_train_ds_cls=shuf_train_ds_cls,
                     train_ds_cls=train_ds_cls,
                     test_ds_cls=test_ds_cls,
                     validation_ds_cls=valid_ds_cls,
                     check_img_path=check_img_path,
                     )

if DATASET == 'A':
    GENERATOR_IMAGE = ['img/17H-0863-1_L0006-new/L0006.png', 'img/17H-0863-1_L0011/L0011.png', 'img/17H-0863-1_L0017/L0017.png']
    ANSWER_IMAGE = ['img/17H-0863-1_L0006-new/L0006_bin.png', 'img/17H-0863-1_L0011/L0011_bin.png', 'img/17H-0863-1_L0017/L0017_bin.png']
    GENERATOR_IMAGE_NAME = ['generator_L0006', 'generator_L0011', 'generator_L0017']
    HLS_IMAGE = ['img/17H-0863-1_L0006-new/L0006_brank_otsu_l.png', 'img/17H-0863-1_L0011/L0011_brank_otsu_l.png', 'img/17H-0863-1_L0017/L0017_brank_otsu_l.png']
elif DATASET == 'B':
    GENERATOR_IMAGE = ['img/17H-0863-1_L0003/L0003.png', 'img/17H-0863-1_L0005/L0005.png', 'img/17H-0863-1_L0009/L0009.png']
    ANSWER_IMAGE = ['img/17H-0863-1_L0003/L0003_bin.png', 'img/17H-0863-1_L0005/L0005_bin.png', 'img/17H-0863-1_L0009/L0009_bin.png']
    GENERATOR_IMAGE_NAME = ['generator_L0003', 'generator_L0005', 'generator_L0009']
    HLS_IMAGE = ['img/17H-0863-1_L0003/L0003_brank_otsu_l.png', 'img/17H-0863-1_L0005/L0005_brank_otsu_l.png', 'img/17H-0863-1_L0009/L0009_brank_otsu_l.png']
elif DATASET == 'C':
    GENERATOR_IMAGE = ['img/17H-0863-1_L0002-old/L0002.png', 'img/17H-0863-1_L0008/L0008.png', 'img/17H-0863-1_L0013/L0013.png']
    ANSWER_IMAGE = ['img/17H-0863-1_L0002-old/L0002_bin.png', 'img/17H-0863-1_L0008/L0008_bin.png', 'img/17H-0863-1_L0013/L0013_bin.png']
    GENERATOR_IMAGE_NAME = ['generator_L0002', 'generator_L0008', 'generator_L0013']
    HLS_IMAGE = ['img/17H-0863-1_L0002-old/L0002_brank_otsu_l.png', 'img/17H-0863-1_L0008/L0008_brank_otsu_l.png', 'img/17H-0863-1_L0013/L0013_brank_otsu_l.png']
elif DATASET == 'D':
    GENERATOR_IMAGE = ['img/17H-0863-1_L0007/L0007.png', 'img/17H-0863-1_L0012/L0012.png', 'img/17H-0863-1_L0014/L0014.png']
    ANSWER_IMAGE = ['img/17H-0863-1_L0007/L0007_bin.png', 'img/17H-0863-1_L0012/L0012_bin.png', 'img/17H-0863-1_L0014/L0014_bin.png']
    GENERATOR_IMAGE_NAME = ['generator_L0007', 'generator_L0012', 'generator_L0014']
    HLS_IMAGE = ['img/17H-0863-1_L0007/L0007_brank_otsu_l.png', 'img/17H-0863-1_L0012/L0012_brank_otsu_l.png', 'img/17H-0863-1_L0014/L0014_brank_otsu_l.png']
elif DATASET == 'E':
    GENERATOR_IMAGE = ['img/17H-0863-1_L0004/L0004.png', 'img/17H-0863-1_L0015/L0015.png', 'img/17H-0863-1_L0016/L0016.png']
    ANSWER_IMAGE = ['img/17H-0863-1_L0004/L0004_bin.png', 'img/17H-0863-1_L0015/L0015_bin.png', 'img/17H-0863-1_L0016/L0016_bin.png']
    GENERATOR_IMAGE_NAME = ['generator_L0004', 'generator_L0015', 'generator_L0016']
    HLS_IMAGE = ['img/17H-0863-1_L0004/L0004_brank_otsu_l.png', 'img/17H-0863-1_L0015/L0015_brank_otsu_l.png', 'img/17H-0863-1_L0016/L0016_brank_otsu_l.png']
elif DATASET == 'F':
    GENERATOR_IMAGE = ['img/17H-0863-1_L0001/L0001.png', 'img/17H-0863-1_L0010/L0010.png', 'img/17H-0863-1_L0018/L0018.png']
    ANSWER_IMAGE = ['img/17H-0863-1_L0001/L0001_bin.png', 'img/17H-0863-1_L0010/L0010_bin.png', 'img/17H-0863-1_L0018/L0018_bin.png']
    GENERATOR_IMAGE_NAME = ['generator_L0001', 'generator_L0010', 'generator_L0018']
    HLS_IMAGE = ['img/17H-0863-1_L0001/L0001_brank_otsu_l.png', 'img/17H-0863-1_L0010/L0010_brank_otsu_l.png', 'img/17H-0863-1_L0018/L0018_brank_otsu_l.png']

# ========== 学習 ==========
if TASK == 'train':
    nfunc.exe_manage('START', EXE_MANAGE_CSV, MAINNAME, DATASET, TASK)
    if args[6] is None:
        notice('TRAINING START: ' + MAINNAME + '-' + DATASET)
    else:
        notice('TRAINING START: [' + args[6] + '] ' + MAINNAME + '-' + DATASET)
    # ===================================================
    import time
    for loop in range(10):
        try:
            fit_cls.fit(
                end_epoch=END_EPOCH,
                device_list=DEVICE_LIST,
                ckpt_step=CHECK_STEP,
            )
        except Exception as e:
            if args[6] is None:
                notice('!!!!! ERROR END !!!!!: ' + MAINNAME + '-' + DATASET)
            else:
                notice('!!!!! ERROR END !!!!!: [' + args[6] + '] ' + MAINNAME + '-' + DATASET)
            print(traceback.format_exc())
            if loop == 9:
                slack_notice(traceback.format_exc())
                sys.exit(1)
            else:
                time.sleep(10)
        else:
            if args[6] is None:
                notice('TRAINING END: ' + MAINNAME + '-' + DATASET)
            else:
                notice('TRAINING END: [' + args[6] + '] ' + MAINNAME + '-' + DATASET)
            break
        finally:
            nfunc.exe_manage('STAY', EXE_MANAGE_CSV, MAINNAME, DATASET, TASK)
    # ===================================================

    nfunc.exe_manage('END', EXE_MANAGE_CSV, MAINNAME, DATASET, TASK)
# ========== 評価 ==========
elif TASK in ['eva', 'eva_only-test', 'eva_only-train', 'eva_only-valid', 'eva_all']:
# elif TASK == 'eva' or TASK =='eva_train-ds' or T:
    nfunc.exe_manage('START', EXE_MANAGE_CSV, MAINNAME, DATASET, 'eva')
    train_check = True if TASK in ['eva_only-train', 'eva_all'] else False
    test_check = True if TASK in ['eva', 'eva_only-test', 'eva_all'] else False
    valid_check = True if TASK in ['eva', 'eva_only-valid', 'eva_all'] else False
    eva_csv_name = 'step_accuracy.csv' if TASK == 'eva' else TASK + '.csv'

    eva_cls = EvaCkptAcc(net_cls=net_cls,
                         path_cls=path_cls,
                         train_ds_cls=train_ds_cls,
                         test_ds_cls=test_ds_cls,
                         validation_ds_cls=valid_ds_cls,
                         check_img_path=check_img_path,
                         eva_csv_name=eva_csv_name,
                         )
    if args[6] is None:
        notice('EVALUATION START: ' + MAINNAME + '-' + DATASET)
    else:
        notice('EVALUATION START: [' + args[6] + '] ' + MAINNAME + '-' + DATASET)

    # ===================================================
    import time
    for loop in range(10):
        try:
            eva_cls.run(end_epoch=END_EPOCH,
                        device_list=DEVICE_LIST,
                        train_check=train_check,
                        test_check=test_check,
                        valid_check=valid_check,
                        )
        except Exception as e:
            if args[6] is None:
                notice('!!!!! ERROR END !!!!!: ' + MAINNAME + '-' + DATASET)
            else:
                notice('!!!!! ERROR END !!!!!: [' + args[6] + '] ' + MAINNAME + '-' + DATASET)
            print(traceback.format_exc())
            if loop == 9:
                slack_notice(traceback.format_exc())
                sys.exit(1)
            else:
                time.sleep(10)
        else:
            if args[6] is None:
                notice('TRAINING END: ' + MAINNAME + '-' + DATASET)
            else:
                notice('TRAINING END: [' + args[6] + '] ' + MAINNAME + '-' + DATASET)
            break
        finally:
            nfunc.exe_manage('STAY', EXE_MANAGE_CSV, MAINNAME, DATASET, 'eva')
    # ===================================================
    nfunc.exe_manage('END', EXE_MANAGE_CSV, MAINNAME, DATASET, 'eva')

    PlotFunc.multi_csv_plot(
        csv_path=path_cls.make_csv_path('step_accuracy.csv'),
        title=OUT_ROOT_FOLDER + '/[ALL]_' + os.path.basename(OUT_ROOT_FOLDER),
        label=DATASET,
        items=['train_accuracy', 'test_accuracy', 'validation_accuracy'],
        ylim=[0.9, 1]
    )

    PlotFunc.multi_csv_plot(
        csv_path=path_cls.make_csv_path('step_accuracy.csv'),
        title=OUT_ROOT_FOLDER + '/[TRAIN_SP]_' + os.path.basename(OUT_ROOT_FOLDER),
        label=DATASET,
        items=['train_accuracy', 'train_weight_accracy', 'train_weight_only_accracy', 'train_boundary_removal_accracy',
               'train_boundary_accracy'],
        ylim=[0.6, 1]
    )

    PlotFunc.multi_csv_plot(
        csv_path=path_cls.make_csv_path('step_accuracy.csv'),
        title=OUT_ROOT_FOLDER + '/[TEST_SP]_' + os.path.basename(OUT_ROOT_FOLDER),
        label=DATASET,
        items=['test_accuracy', 'test_weight_accracy', 'test_weight_only_accracy', 'test_boundary_removal_accracy',
               'test_boundary_accracy'],
        ylim=[0.6, 1]
    )

    PlotFunc.multi_csv_plot(
        csv_path=path_cls.make_csv_path('step_accuracy.csv'),
        title=OUT_ROOT_FOLDER + '/[VALID_SP]_' + os.path.basename(OUT_ROOT_FOLDER),
        label=DATASET,
        items=['validation_accuracy', 'valid_weight_accracy', 'valid_weight_only_accracy', 'valid_boundary_removal_accracy',
               'valid_boundary_accracy'],
        ylim=[0.6, 1]
    )


    # ========== save model ==========
    gen_model_path = path_cls.make_model_path(MAINNAME + '-' + DATASET + '_gen.h5')
    # bin_model_path = path_cls.make_model_path(MAINNAME + '-' + DATASET + '_bin.h5')
    nfunc.save_best_generator_model(net_cls=net_cls, path_cls=path_cls, path=gen_model_path)
    # nfunc.save_best_binary_model(net_cls=net_cls, path_cls=path_cls, path=bin_model_path)

    # ========== 生成速度計測 ==========
    gen_cls = ImageGenerator(Generator_model=gen_model_path,
                             model_h=IMAGE_HEIGHT,
                             model_w=IMAGE_WIDTH,
                             fin_activate=FIN_ACTIVATE,
                             padding=net_cls.get_padding(),
                             standardization_csv_path=STANDARDIZATION_CSV_PATH,)

    for i in range(len(GENERATOR_IMAGE)):
        gen_cls.run(img_path=DATASET_PATH + '/' + GENERATOR_IMAGE[i],
                    out_path=OUT_ROOT_FOLDER + '/' + GENERATOR_IMAGE_NAME[i] + '.png',
                    hls_path=DATASET_PATH + '/' + HLS_IMAGE[i],
                    time_out_path=path_cls.make_csv_path('Generator_time_'+GENERATOR_IMAGE_NAME[i]+'.csv'))

elif TASK == 'gen_img':
    gen_model_path = path_cls.make_model_path(MAINNAME + '-' + DATASET + '_gen.h5')
    # bin_model_path = path_cls.make_model_path(MAINNAME + '-' + DATASET + '_bin.h5')
    nfunc.save_best_generator_model(net_cls=net_cls, path_cls=path_cls, path=gen_model_path)

    gen_cls = ImageGenerator(Generator_model=gen_model_path,
                             model_h=IMAGE_HEIGHT,
                             model_w=IMAGE_WIDTH,
                             fin_activate=FIN_ACTIVATE,
                             padding=net_cls.get_padding(),
                             standardization_csv_path=STANDARDIZATION_CSV_PATH,)

    # print('Z:/hayakawa/work20/dataset2/img' + '/' + HLS_IMAGE[0])
    # sys.exit()

    for i in range(len(GENERATOR_IMAGE)):
        gen_cls.run(img_path=DATASET_PATH + '/' + GENERATOR_IMAGE[i],
                    out_path=OUT_ROOT_FOLDER + '/' + GENERATOR_IMAGE_NAME[i] + '.png',
                    hls_path='Z:/hayakawa/work20/dataset2' + '/' + HLS_IMAGE[i],
                    time_out_path=path_cls.make_csv_path('Generator_time_'+GENERATOR_IMAGE_NAME[i]+'.csv'))

elif TASK == 'eva_ckpt':
    from EvaCkpt import EvaCkpt
    GENERATOR_IMAGE = [DATASET_PATH + '/' + path for path in GENERATOR_IMAGE]
    ANSWER_IMAGE = [DATASET_PATH + '/' + path for path in ANSWER_IMAGE]

    evackpt_cls = EvaCkpt(
        net_cls=net_cls,
        path_cls=path_cls,
        exe_file=work_dirname + '/PatternEva_1130.exe',
        model_hw=[IMAGE_HEIGHT, IMAGE_WIDTH],
        fin_activate=FIN_ACTIVATE,
    )

    evackpt_cls.run(img_path_set=GENERATOR_IMAGE, ans_path_set=ANSWER_IMAGE)

elif TASK == 'plot':
    PlotFunc.multi_csv_plot(
        csv_path=path_cls.make_csv_path('step_accuracy.csv'),
        title=OUT_ROOT_FOLDER + '/[ALL]_' + os.path.basename(OUT_ROOT_FOLDER),
        label=DATASET,
        items=['train_accuracy', 'test_accuracy', 'validation_accuracy'],
        ylim=[0.9, 1]
    )


    PlotFunc.multi_csv_plot(
        csv_path=path_cls.make_csv_path('step_accuracy.csv'),
        title=OUT_ROOT_FOLDER + '/[weight]_' + os.path.basename(OUT_ROOT_FOLDER),
        label=DATASET,
        items=['train_weight_only_accracy', 'test_weight_only_accracy', 'valid_weight_only_accracy'],
        ylim=[0.9, 1]
    )

    PlotFunc.multi_csv_plot(
        csv_path=path_cls.make_csv_path('step_accuracy.csv'),
        title=OUT_ROOT_FOLDER + '/[test_N_vs_W]_' + os.path.basename(OUT_ROOT_FOLDER),
        label=DATASET,
        items=['test_accuracy', 'test_weight_only_accracy'],
        ylim=[0.9, 1]
    )

    PlotFunc.multi_csv_plot(
        csv_path=path_cls.make_csv_path('step_accuracy.csv'),
        title=OUT_ROOT_FOLDER + '/[valid_N_vs_W]_' + os.path.basename(OUT_ROOT_FOLDER),
        label=DATASET,
        items=['validation_accuracy', 'valid_weight_only_accracy'],
        ylim=[0.9, 1]
    )

elif TASK == 'other':
    path_list = path_cls.search_best_path(filename=path_cls.ckpt_step_name)
    # if path_list:
    # print(path_list)
    # root = 'Z:/hayakawa/binary/20201218/unet_cw3-d01-1_fin-none_logit-true_thr-d0/unet_cw3-d01-1_fin-none_logit-true_thr-d0-A/CheckpointModel/ckpt-step1313280-74'
    root, ext = os.path.splitext(path_list[0])
    net_cls.ckpt_restore(path=root)
    gen_model_path = path_cls.make_model_path(MAINNAME + '-' + DATASET + '_gen.h5')
    # nfunc.save_best_generator_model(net_cls=net_cls, path_cls=path_cls, path=gen_model_path)
    net_cls.get_generator().save(gen_model_path)

    gen_cls = ImageGenerator(Generator_model=gen_model_path,
                             model_h=IMAGE_HEIGHT,
                             model_w=IMAGE_WIDTH,
                             fin_activate=FIN_ACTIVATE,
                             padding=net_cls.get_padding(),
                             standardization_csv_path=STANDARDIZATION_CSV_PATH,)

    for i in range(len(GENERATOR_IMAGE)):
        gen_cls.run(img_path=DATASET_PATH + '/' + GENERATOR_IMAGE[i],
                    out_path=OUT_ROOT_FOLDER + '/' + GENERATOR_IMAGE_NAME[i] + '.png',
                    hls_path='Z:/hayakawa/work20/dataset2' + '/' + HLS_IMAGE[i],
                    time_out_path=path_cls.make_csv_path('Generator_time_'+GENERATOR_IMAGE_NAME[i]+'.csv'))

