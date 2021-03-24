import os
import glob
from tqdm import tqdm
import csv
import subprocess
import cv2
import shutil
import sys
import math
import numpy as np
from natsort import natsorted

# ANS_PATHSET = [
#     'E:/work/myTensor/dataset2/img/17H-0863-1_L0001/L0001_bin.png',
#     'E:/work/myTensor/dataset2/img/17H-0863-1_L0002-old/L0002_bin.png',
#     'E:/work/myTensor/dataset2/img/17H-0863-1_L0003/L0003_bin.png',
#     'E:/work/myTensor/dataset2/img/17H-0863-1_L0004/L0004_bin.png',
#     'E:/work/myTensor/dataset2/img/17H-0863-1_L0005/L0005_bin.png',
#     'E:/work/myTensor/dataset2/img/17H-0863-1_L0006-new/L0006_bin.png',
#     'E:/work/myTensor/dataset2/img/17H-0863-1_L0007/L0007_bin.png',
#     'E:/work/myTensor/dataset2/img/17H-0863-1_L0008/L0008_bin.png',
#     'E:/work/myTensor/dataset2/img/17H-0863-1_L0009/L0009_bin.png',
#     'E:/work/myTensor/dataset2/img/17H-0863-1_L0010/L0010_bin.png',
#     'E:/work/myTensor/dataset2/img/17H-0863-1_L0011/L0011_bin.png',
#     'E:/work/myTensor/dataset2/img/17H-0863-1_L0012/L0012_bin.png',
#     'E:/work/myTensor/dataset2/img/17H-0863-1_L0013/L0013_bin.png',
#     'E:/work/myTensor/dataset2/img/17H-0863-1_L0014/L0014_bin.png',
#     'E:/work/myTensor/dataset2/img/17H-0863-1_L0015/L0015_bin.png',
#     'E:/work/myTensor/dataset2/img/17H-0863-1_L0016/L0016_bin.png',
#     'E:/work/myTensor/dataset2/img/17H-0863-1_L0017/L0017_bin.png',
#     'E:/work/myTensor/dataset2/img/17H-0863-1_L0018/L0018_bin.png',
# ]C:\Users\hayakawa\work\mytensor\dataset2

ANS_PATHSET = [
    'C:/Users/hayakawa/work/myTensor/dataset2/img/17H-0863-1_L0001/L0001_bin.png',
    'C:/Users/hayakawa/work/myTensor/dataset2/img/17H-0863-1_L0002-old/L0002_bin.png',
    'C:/Users/hayakawa/work/myTensor/dataset2/img/17H-0863-1_L0003/L0003_bin.png',
    'C:/Users/hayakawa/work/myTensor/dataset2/img/17H-0863-1_L0004/L0004_bin.png',
    'C:/Users/hayakawa/work/myTensor/dataset2/img/17H-0863-1_L0005/L0005_bin.png',
    'C:/Users/hayakawa/work/myTensor/dataset2/img/17H-0863-1_L0006-new/L0006_bin.png',
    'C:/Users/hayakawa/work/myTensor/dataset2/img/17H-0863-1_L0007/L0007_bin.png',
    'C:/Users/hayakawa/work/myTensor/dataset2/img/17H-0863-1_L0008/L0008_bin.png',
    'C:/Users/hayakawa/work/myTensor/dataset2/img/17H-0863-1_L0009/L0009_bin.png',
    'C:/Users/hayakawa/work/myTensor/dataset2/img/17H-0863-1_L0010/L0010_bin.png',
    'C:/Users/hayakawa/work/myTensor/dataset2/img/17H-0863-1_L0011/L0011_bin.png',
    'C:/Users/hayakawa/work/myTensor/dataset2/img/17H-0863-1_L0012/L0012_bin.png',
    'C:/Users/hayakawa/work/myTensor/dataset2/img/17H-0863-1_L0013/L0013_bin.png',
    'C:/Users/hayakawa/work/myTensor/dataset2/img/17H-0863-1_L0014/L0014_bin.png',
    'C:/Users/hayakawa/work/myTensor/dataset2/img/17H-0863-1_L0015/L0015_bin.png',
    'C:/Users/hayakawa/work/myTensor/dataset2/img/17H-0863-1_L0016/L0016_bin.png',
    'C:/Users/hayakawa/work/myTensor/dataset2/img/17H-0863-1_L0017/L0017_bin.png',
    'C:/Users/hayakawa/work/myTensor/dataset2/img/17H-0863-1_L0018/L0018_bin.png',
]

EXE_FILE = 'PatternEva_1130.exe'
CHECK_PATH = 'Z:/hayakawa/binary/20200927'  # 絶対パス


def check_eveluate(path, csv_name, file_count):
    index = 0
    total_division = 0
    total_short = 0
    total_short_area = 0
    total_break = 0
    total_break_area = 0
    total_break_same = 0
    total_break_same_area = 0
    if os.path.exists(path + '/' + csv_name):
        os.remove(path + '/' + csv_name)
    with open(path + '/Evaluate.txt', 'r') as fp:
        with open(path + '/' + csv_name, 'w') as csv_file:

            writer = csv.writer(csv_file, lineterminator='\n')
            file = fp.readlines()
            for img_number in range(file_count):
                vals = [img_number]

                while True:
                    Str = file[index]
                    if Str[0] == '教':
                        break
                    else:
                        index += 1

                # 教師画像 パターン数
                Str = file[index]
                start = Str.find('数') + 2
                end = Str.find('個')
                val = Str[start: end]
                vals.append(val)
                index += 1

                # 推論2値画像 パターン数
                Str = file[index]
                start = Str.find('数') + 2
                end = Str.find('個')
                val = Str[start: end]
                vals.append(val)
                index += 1

                # 1対1 対応パターン数
                Str = file[index]
                start = Str.find('数') + 2
                end = Str.find('個')
                val = Str[start: end]
                vals.append(val)

                start = Str.find('個') + 2
                end = Str.find('%')
                val = Str[start: end]
                vals.append(val)
                index += 1

                # 同一電位パターン分断数
                Str = file[index]
                start = Str.find('数') + 2
                end = len(Str) - 1
                val = Str[start: end]
                vals.append(val)
                total_division += int(Str[start: end])
                index += 1

                # Short positions
                Str = file[index]
                start = Str.find(':') + 1
                end = len(Str) - 1
                val = Str[start: end]
                vals.append(val)
                total_short += int(Str[start: end])
                index += 1

                # Short positions area
                val = 0
                while True:
                    Str = file[index]
                    if '{' not in Str:
                        break
                    else:
                        val += 1
                        index += 1
                vals.append(val)
                total_short_area += val

                # Break positions
                Str = file[index]
                start = Str.find(':') + 1
                end = len(Str) - 1
                val = Str[start: end]
                vals.append(val)
                total_break += int(Str[start: end])
                index += 1

                # Break positions Point
                val = 0
                while True:
                    Str = file[index]
                    if '{' not in Str:
                        break
                    else:
                        val += 1
                        index += 1
                vals.append(val)
                total_break_area += val

                # Break positions of same potential pattern
                Str = file[index]
                start = Str.find(':') + 1
                end = len(Str) - 1
                val = Str[start: end]
                vals.append(val)
                total_break_same += int(Str[start: end])
                index += 1

                # Break positions of same potential pattern area
                # index += 2
                val = 0
                while True:
                    Str = file[index]
                    if '{' not in Str:
                        break
                    else:
                        val += 1
                        index += 1
                vals.append(val)
                total_break_same_area += val

                writer.writerow(vals)
            # === 繰り返しここまで ===
            writer.writerow(
                ['TOTAL', '', '', '', '', str(total_division), str(total_short), str(total_short_area),
                 str(total_break), str(total_break_area), str(total_break_same), str(total_break_same_area)])

            # if not os.path.exists(path + '/'+'PatternResult.csv'):
            #     with open(path+ '/'+'PatternResult.csv', 'w') as total_file:
            #         writer_total = csv.writer(total_file, lineterminator='\n')
            #         writer_total.writerow(['total_division', 'total_short_area',
            #                             'total_break_area', 'total_break_same_area'])

            # with open(path + '/'+'PatternResult.csv', 'a') as total_file:
            # writer_total = csv.writer(total_file, lineterminator='\n')
            writer.writerow(['total_division', 'total_short_area',
                             'total_break_area', 'total_break_same_area'])
            writer.writerow(
                [total_division, total_short_area, total_break_area, total_break_same_area])
            writer.writerow(['short', 'break+break_same', 'total'])
            writer.writerow([total_short_area, total_break_area + total_break_same_area,
                             total_short_area + total_break_area + total_break_same_area])

    return [total_division, total_short, total_short_area, total_break, total_break_area, total_break_same,
            total_break_same_area]


# out_files = glob.glob(LEARN_BOND_PATH + '/Output_*.png')
# out_files = natsorted( out_files )
# LEARN_BOND_PATH = base_path + '/' + IMAGE_FOLDER + '/learn_bond'

# directorys_1 = glob.glob(CHECK_PATH + '/*')
# # directorys_2 = []
# for j in range(len(directorys_1)):
#     print('progress: ' + str(j+1) + ' / ' + str(len(directorys_1)) )
#     directorys_2 = glob.glob(directorys_1[j] + '/*')
#     if not directorys_2:
#         continue
# directorys_2.append(dirs)

# for index in tqdm(range(file_count), desc='Processed check learn'):
args = sys.argv
directorys_1 = args[1]
directorys_2 = glob.glob(directorys_1 + '/**/')
if not directorys_2:
    sys.exit()
all_ds_total = np.array([0, 0, 0, 0, 0, 0, 0, ])
for i in tqdm(range(len(directorys_2)), desc='Processed check pattern'):
    if os.path.exists(directorys_2[i] + '/' + 'DataSetPatternResult.csv'):
        os.remove(directorys_2[i] + '/' + 'DataSetPatternResult.csv')
    # PatternEva.exeを実行していく
    # directorys_3 = directorys_2[i]
    out_pathset = glob.glob(directorys_2[i] + '/generator_L00??.png')
    ds_total = np.array([0, 0, 0, 0, 0, 0,
                         0, ])  # ds_total_division, ds_total_short, ds_total_short_area, ds_total_break, ds_total_break_area, ds_total_break_same, ds_total_break_same_area
    # base_path = directorys_2[i]
    for out_path in out_pathset:  # A, B, C,...
        # Evaluate.txtがあれば削除しておく
        if os.path.exists(directorys_2[i] + '/Evaluate.txt'):
            os.remove(directorys_2[i] + '/Evaluate.txt')
        # out_pathから使用する解答画像を求める
        s = out_path.find('L00')
        img_n = int(out_path[s + 1: s + 5])
        img_name = out_path[s: s + 5]
        ans_path = ANS_PATHSET[img_n - 1]
        # 処理済みなら飛ばす
        # if os.path.exists(directorys_2[i] + '/'+ 'L'+str(img_n)+'_PatternResult.csv'):
        #     continue
        # 画像を4分割

        subprocess.call(EXE_FILE + ' ' + ans_path + ' ' + out_path)

        total_result = check_eveluate(path=directorys_2[i] + '/', csv_name='L' + str(img_n) + '_PatternResult.csv',
                                      file_count=1)
        ds_total += np.array(
            total_result)  # ds_total_division, ds_total_short, ds_total_short_area, ds_total_break, ds_total_break_area, ds_total_break_same, ds_total_break_same_area
        all_ds_total += np.array(total_result)

        if not os.path.exists(directorys_2[i] + '/' + 'DataSetPatternResult.csv'):
            with open(directorys_2[i] + '/' + 'DataSetPatternResult.csv', 'w') as total_file:
                writer_total = csv.writer(total_file, lineterminator='\n')
                writer_total.writerow(['', 'total_division', 'total_short_area',
                                       'total_break_area + total_break_same_area', 'total_break_area',
                                       'total_break_same_area'])

        with open(directorys_2[i] + '/' + 'DataSetPatternResult.csv', 'a') as total_file:
            writer_total = csv.writer(total_file, lineterminator='\n')
            writer_total.writerow(
                ['L' + str(img_n), total_result[0], total_result[2], total_result[4] + total_result[6], total_result[4],
                 total_result[6]])

    with open(directorys_2[i] + '/' + 'DataSetPatternResult.csv', 'a') as total_file:
        writer_total = csv.writer(total_file, lineterminator='\n')
        writer_total.writerow(['total_division', 'total_short_area',
                               'total_break_area + total_break_same_area', 'total_break_area', 'total_break_same_area'])
        writer_total.writerow([ds_total[0], ds_total[2], ds_total[4] + ds_total[6], ds_total[4], ds_total[6]])
        writer_total.writerow(['short', 'break+break_same', 'total'])
        writer_total.writerow([ds_total[2], ds_total[4] + ds_total[6], ds_total[2] + ds_total[4] + ds_total[6]])

with open(directorys_1 + '/' + 'TotalPatternResult.csv', 'w') as total_file:
    writer_total = csv.writer(total_file, lineterminator='\n')
    writer_total.writerow(['total_division', 'total_short_area',
                           'total_break_area + total_break_same_area', 'total_break_area', 'total_break_same_area'])
    writer_total.writerow(
        [all_ds_total[0], all_ds_total[2], all_ds_total[4] + all_ds_total[6], all_ds_total[4], all_ds_total[6]])
    writer_total.writerow(['short', 'break+break_same', 'total'])
    writer_total.writerow(
        [all_ds_total[2], all_ds_total[4] + all_ds_total[6], all_ds_total[2] + all_ds_total[4] + all_ds_total[6]])
    # if not os.path.exists(directorys_2[j] + '/'+'TotalPatternResult.csv'):
    #     with open(directorys_2[j] + '/'+'TotalPatternResult.csv', 'w') as total_file:
    #         writer_total = csv.writer(total_file, lineterminator='\n')
    #         writer_total.writerow(['', 'total_division', 'total_short_area',
    #                                'total_break_area + total_break_same_area', 'total_break_area', 'total_break_same_area'])

    # with open(directorys_2[j] + '/'+'TotalPatternResult.csv', 'a') as total_file:
    #     writer_total = csv.writer(total_file, lineterminator='\n')
    #     writer_total.writerow(
    #         ['L'+str(img_n), ds_total[0], ds_total[2], ds_total[4] + ds_total[6], ds_total[4], ds_total[6]])
    # [total_division, total_short_area, total_break_area, total_break_same_area])
    # writer_total.writerow(['short', 'break+break_same', 'total'])
    # writer_total.writerow([ds_total[2], ds_total[4]+ds_total[6],
    #                     ds_total[2] + ds_total[4] + ds_total[6]])
    # writer_total.writerow([total_short_area, total_break_area+total_break_same_area,
    #                     total_short_area + total_break_area + total_break_same_area])


