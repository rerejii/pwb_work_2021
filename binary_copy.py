import numpy as np
import pandas as pd
import shutil
from distutils.dir_util import copy_tree
import glob
from natsort import natsorted
import os


# copy_tree("./test1", "./test2")

# ほぼ固定
original_path = 'Z:/hayakawa/binary'
copy_to_path = 'G:/binary'
copy_target_set = ['CsvDatas', 'StepLog']

# 修正項目
root_path = '20210303/unet_rgb_otsu_loop'
csv_name = 'eva_all.csv'
epoch = 80

original_path = original_path + '/' + root_path
copy_to_path = copy_to_path + '/' + root_path

# 処理開始
# フォルダ作成
os.makedirs(copy_to_path, exist_ok=True)
# csvコピー
csv_set = glob.glob(original_path + '/*.csv')
for csv_path in csv_set:
    shutil.copy(csv_path, copy_to_path + '/' + os.path.basename(csv_path))

ds_pathset = glob.glob(original_path + '/*-*')
ds_pathset = natsorted(ds_pathset)
for ds_path in ds_pathset:  # ds単位
    print(ds_path)
    # ds_path <- Z:/hayakawa/binary/20210227/unet_rgb_otsu_loop\unet_rgb_otsu_loop-A

    # フォルダ作成
    os.makedirs(copy_to_path + '/' + os.path.basename(ds_path), exist_ok=True)

    # 画像コピー
    png_set = glob.glob(ds_path + '/*.png')
    for png_path in png_set:
        shutil.copy(png_path, copy_to_path + '/' + os.path.basename(ds_path) + '/' + os.path.basename(png_path))

    # csvコピー
    csv_set = glob.glob(ds_path + '/*.csv')
    for csv_path in csv_set:
        shutil.copy(csv_path, copy_to_path + '/' + os.path.basename(ds_path) + '/' + os.path.basename(csv_path))

    # フォルダ単位のコピー
    for copy_target in copy_target_set:  # フォルダ単位
        copy_tree(ds_path + '/' + copy_target, copy_to_path + '/' + os.path.basename(ds_path) + '/' + copy_target)

    # 最良モデル取得
    csv_path = ds_path + '/CsvDatas/' + csv_name
    df = pd.read_csv(csv_path)
    max_test_acc_i = df['test_accuracy'][:epoch].idxmax()

    # ベストモデルをコピー
    before = ds_path + '/CheckpointModel'
    after = copy_to_path + '/' + os.path.basename(ds_path) + '/CheckpointModel'
    os.makedirs(after, exist_ok=True)
    ckpt_path_set = glob.glob(before + '/ckpt-step' + str(df['step'][max_test_acc_i]) + '*')
    for ckpt_path in ckpt_path_set:
        shutil.copy(ckpt_path, after + '/' + os.path.basename(ckpt_path))
    # shutil.copy(png_path, copy_to_path + '/' + os.path.basename(ds_path) + '/' + os.path.basename(png_path))







