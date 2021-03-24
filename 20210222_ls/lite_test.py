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
MAINNAME = 'unet'
# importする net cls
IMPORT_NET_CLS = 'unet_0001_sigmoid_detac'
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
CYCLE_LENGHT = 4  # データ読み込み処理オーバーラップ数
SHUF_LEARN_BATCH_SIZE = 20
LEARN_BATCH_SIZE = 50
VALIDATION_BATCH_SIZE = 50
TEST_BATCH_SIZE = 50
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256
SUFFLE_BUFFER_SIZE = 100000
OUTPUT_CHANNELS = 1
CHECK_STEP = None
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
STANDARDIZATION_CSV_PATH = DATASET_PATH + '/std/std-' + DATASET + '.csv'
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
                                )
train_ds_cls = DataManager(tfrecord_path=path_cls.get_train_ds_path(),
                           img_root=IMAGE_ROOT_PATH,
                           batch_size=LEARN_BATCH_SIZE,
                           net_cls=net_cls,
                           data_n=df.at[0, 'total_learn_data'],
                           )
test_ds_cls = DataManager(tfrecord_path=path_cls.get_test_ds_path(),
                          img_root=IMAGE_ROOT_PATH,
                          batch_size=TEST_BATCH_SIZE,
                          net_cls=net_cls,
                          data_n=df.at[0, 'total_test_data'],
                          )
valid_ds_cls = DataManager(tfrecord_path=path_cls.get_validation_ds_path(),
                          img_root=IMAGE_ROOT_PATH,
                          batch_size=VALIDATION_BATCH_SIZE,
                          net_cls=net_cls,
                          data_n=df.at[0, 'total_valid_data'],
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

# ========== 学習＆評価 ==========
# fit_cls.fit(
#     end_epoch=END_EPOCH,
#     device_list=DEVICE_LIST,
#     ckpt_step=CHECK_STEP,
# )

# modelpath = path_cls.make_model_path(MAINNAME + '-' + DATASET + '.h5')
# print(modelpath)
# model = tf.keras.models.load_model(modelpath)
# nfunc.save_best_generator_model(net_cls=net_cls, path_cls=path_cls, path=modelpath)
path_list = path_cls.search_best_path(filename=path_cls.ckpt_step_name)
if path_list:
    root, ext = os.path.splitext(path_list[0])
    net_cls.ckpt_restore(path=root)
gen = net_cls.get_generator()

# concrete_func = model.signatures[
#   tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
# concrete_func.inputs[0].set_shape([1, 256, 256, 3])
# converter = TFLiteConverter.from_concrete_functions([concrete_func])

converter = tf.lite.TFLiteConverter.from_keras_model(gen)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]
tflite_model = converter.convert()
open("converted_model.tflite", "wb").write(tflite_model)
interpreter_fp16 = tf.lite.Interpreter(model_path="converted_model.tflite")
interpreter_fp16.allocate_tensors()
img_byte = tf.io.read_file('Sample_X002_Y019.png')
in_img = tf.image.decode_png(img_byte, channels=3)
in_img = in_img / 255
H, W, C = in_img.shape
in_img = tf.reshape(in_img, [1, H, W, C])
# print(in_img.shape)
# concrete_func = root.f.get_concrete_function(input_data)
# sys.exit()

# in_img = reshape(im_img, [])
# in_img = tf.constant([in_img])

input_details = interpreter_fp16.get_input_details()
input_shape = input_details[0]['shape']
print(input_shape)
interpreter_fp16.resize_tensor_input(0, [1, 256, 256, 3], strict=True)
interpreter_fp16.allocate_tensors()
input_details = interpreter_fp16.get_input_details()
input_shape = input_details[0]['shape']
print(input_shape)

output_details = interpreter_fp16.get_output_details()
output_shape = output_details[0]['shape']
print(output_shape)

input_index = interpreter_fp16.get_input_details()[0]["index"]
# print(interpreter_fp16.get_input_details())
output_index = interpreter_fp16.get_output_details()[0]["index"]
# print(interpreter_fp16.get_output_details())
# # in_img = np.ones([1,1,1,3], dtype=np.float32)
# print('===========')
interpreter_fp16.set_tensor(input_index, in_img)
output = interpreter_fp16.tensor(output_index)
# print('===========')
interpreter_fp16.invoke()
# print('===========')
print(output())
# print('===========')
out_img = tf.greater_equal(output(), 0.5)
out_img = tf.cast(out_img, tf.uint8)
out_img = tf.multiply(out_img, 255)
out_img = tf.cast(out_img, tf.uint8)
out_img = tf.reshape(out_img, [H, W, 1])
out_byte = tf.image.encode_png(out_img)
tf.io.write_file(filename='out.png', contents=out_byte)



# print(input_index)
# print(output_index)


# out = self.Generator(data, training=False)


# def netdata_from_img(self, data, fin_activate):
#     if fin_activate is 'tanh':
#         return (data / 127.5) - 1
#     if fin_activate is 'sigmoid':
#         return data / 255