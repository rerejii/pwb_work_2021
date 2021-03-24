from PIL import Image
import os
import numpy as np
import tensorflow.compat.v1 as tf
import glob
import shutil
import cv2
import sys
import math
import random
from tqdm import tqdm
import csv

HEIGHT_CROP_SIZE = 256
WIDTH_CROP_SIZE = 256
PADDING_BRANK = 28
SAMPLE_FILE = 'E:/work/myTensor/dataset/img/17H-0863-1_L0002-new/L0002.png'
ANSWER_FILE = 'E:/work/myTensor/dataset/img/17H-0863-1_L0002-new/L0002_bin.png'
IMG_OUTPUT_FOLDER_NAME = 'Img-256-valid'+str(PADDING_BRANK)+'-L2'
IMG_OUTPUT_FOLDER_BASE = '../../dataset2/'

DR_OUTPUT_FOLDER_NAME = 'DR-256-valid'+str(PADDING_BRANK)+'-L2'
DR_OUTPUT_FOLDER_BASE = 'E:/work/myTensor/dataset2/'
SETNAME = ['A', 'B', 'C', 'D']
DATASET = 4

OUTPUT_PROPERTY_FILE = 'Property'
OUTPUT_LEARN_FILE = 'LearnData'
OUTPUT_TEST_FILE = 'TestData'
OUTPUT_TEXT_FILE = 'TotalData'

img_out_folder_name = IMG_OUTPUT_FOLDER_BASE + IMG_OUTPUT_FOLDER_NAME+'/'+IMG_OUTPUT_FOLDER_NAME+'-img'
out_sample_folder = img_out_folder_name+'/'+'Sample'
out_answer_folder = img_out_folder_name+'/'+'Answer'
os.makedirs(out_sample_folder, exist_ok=True)
os.makedirs(out_answer_folder, exist_ok=True)

seed = 0
# set Python random seed
random.seed(seed)
# set NumPy random seed
np.random.seed(seed)

def check_brank(img):
    print('check brank process now')
    height, width, _ = img.shape
    top, bottom, left, right = 0, 0, 0, 0
    top_brank, bottom_brank, left_brank, right_brank = 0, 0, 0, 0

    # top
    for i in range(height):
        read = i
        line = img[read:read + 1, :, :]
        val = np.sum(line)
        if val != 0:
            top = read
            top_brank = read
            break

    # bottom
    for i in range(height):
        read = height - i - 1
        line = img[read:read + 1, :, :]
        val = np.sum(line)
        if val != 0:
            bottom = read
            bottom_brank = height - read - 1
            break

    # left
    for i in range(width):
        read = i
        line = img[:, read:read + 1, :]
        val = np.sum(line)
        if val != 0:
            left = read
            left_brank = read
            break

    # right
    for i in range(width):
        read = width - i - 1
        line = img[:, read:read + 1, :]
        val = np.sum(line)
        if val != 0:
            right = read
            right_brank = width - read - 1
            break

    brank = [top_brank, bottom_brank, left_brank, right_brank]
    print('check brank process end')
    print(brank)
    return brank

seed = 0
# set Python random seed
random.seed(seed)
# set NumPy random seed
np.random.seed(seed)

indata = cv2.imread(SAMPLE_FILE)
img = np.array(indata, dtype=np.uint8)
original_height, original_width, _ = img.shape

top_brank, bottom_brank, left_brank, right_brank = check_brank(img)
sample_data = img[top_brank:original_height - bottom_brank, left_brank:original_width - right_brank, :]
answer_imread = cv2.imread(ANSWER_FILE, 0)[top_brank:original_height - bottom_brank,
                left_brank:original_width - right_brank]
answer_data = np.array(answer_imread, dtype=np.uint8)

brankcut_height, brankcut_width, _ = sample_data.shape

if brankcut_height % 2 != 0:
    height_plus_flag = 1
else:
    height_plus_flag = 0

if brankcut_width % 2 != 0:
    width_plus_flag = 1
else:
    width_plus_flag = 0

cal_height = brankcut_height + height_plus_flag
cal_width = brankcut_width + width_plus_flag

height_total_sheets = math.ceil(cal_height / (HEIGHT_CROP_SIZE))
if height_total_sheets  % 2 == 1:
    height_total_sheets += 1
height_total_len = height_total_sheets * (HEIGHT_CROP_SIZE)
height_extra = height_total_len - cal_height

if height_extra % 2 == 1:
    height_extra += 1

width_total_sheets = math.ceil(cal_width / (WIDTH_CROP_SIZE))
if width_total_sheets  % 2 == 1:
    width_total_sheets += 1
width_total_len = width_total_sheets * (WIDTH_CROP_SIZE)
width_extra = width_total_len - cal_width
if width_extra % 2 == 1:
    width_extra += 1

height_flame = brankcut_height + height_extra + height_plus_flag + (PADDING_BRANK * 2)
width_flame = brankcut_width + width_extra + width_plus_flag + (PADDING_BRANK * 2)

sample_image = np.zeros((height_flame, width_flame, 3), dtype=np.uint8)
answer_image = np.zeros((height_flame, width_flame), dtype=np.uint8)
height_start = int(height_extra / 2) + int( (PADDING_BRANK) )
height_end = int(brankcut_height + (height_extra / 2) + PADDING_BRANK)
width_start = int(width_extra / 2) + int( (PADDING_BRANK) )
width_end = int(brankcut_width + (width_extra / 2) + PADDING_BRANK)

sample_image[height_start: height_end, width_start: width_end, :] = sample_data
answer_image[height_start: height_end, width_start: width_end] = answer_data

process_height, process_width, _ = sample_image.shape

# print(sample_image.shape)
# sys.exit()

process_height_half = int(process_height / 2)
process_width_half = int(process_width / 2)
img = sample_image

height_crop = HEIGHT_CROP_SIZE + (PADDING_BRANK * 2)
width_crop = WIDTH_CROP_SIZE + (PADDING_BRANK * 2)

test_height_start_point = [0, 0, process_height_half - (PADDING_BRANK), process_height_half - (PADDING_BRANK)]
test_width_start_point = [0, process_width_half - (PADDING_BRANK), 0, process_width_half - (PADDING_BRANK)]
learn_height_start_point = [process_height_half, process_height_half, 0, 0]
learn_width_start_point = [process_width_half, 0, process_height_half, 0]


test_height_end_point = [process_height_half + (PADDING_BRANK), process_height_half + (PADDING_BRANK),
                    process_height, process_height]
test_width_end_point = [process_width_half + (PADDING_BRANK), process_width,
                   process_width_half + (PADDING_BRANK), process_width]
learn_height_end_point = [process_height, process_height, process_height_half, process_height_half]
learn_width_end_point = [process_width, process_width_half, process_width, process_width_half]

total_session = int((process_height - (PADDING_BRANK * 2)) / (HEIGHT_CROP_SIZE)) * \
                int((process_width - (PADDING_BRANK * 2)) / (WIDTH_CROP_SIZE))

for set in range(DATASET):
    height_test = test_height_end_point[set] - test_height_start_point[set]
    width_test = test_width_end_point[set] - test_width_start_point[set]
    total_learn_data = 0
    total_test_data = 0
    out_folder_name =  DR_OUTPUT_FOLDER_BASE + DR_OUTPUT_FOLDER_NAME+'/'+DR_OUTPUT_FOLDER_NAME+'-'+SETNAME[set]
    out_learn_file_name = out_folder_name +'/'+OUTPUT_LEARN_FILE+'-'+SETNAME[set]+'.tfrecords'
    out_test_file_name = out_folder_name +'/'+OUTPUT_TEST_FILE+'-'+SETNAME[set]+'.tfrecords'
    out_property_file_name = out_folder_name +'/'+OUTPUT_PROPERTY_FILE+'-'+SETNAME[set]+'.tfrecords'
    os.makedirs(out_folder_name, exist_ok=True)
    learn_sample_mask = np.ones(shape=(process_height, process_width, 3), dtype=np.uint8)
    learn_answer_mask = np.ones(shape=(process_height, process_width), dtype=np.uint8)
    test_sample_mask = np.zeros(shape=(process_height, process_width, 3), dtype=np.uint8)
    test_answer_mask = np.zeros(shape=(process_height, process_width), dtype=np.uint8)

    learn_sample_mask[test_height_start_point[set]: test_height_end_point[set], test_width_start_point[set]: test_width_end_point[set], :] = np.zeros(shape=(process_height_half+(PADDING_BRANK), process_width_half+(PADDING_BRANK), 3), dtype=np.uint8)
    learn_answer_mask[test_height_start_point[set]: test_height_end_point[set], test_width_start_point[set]: test_width_end_point[set]] = np.zeros(shape=(process_height_half+(PADDING_BRANK), process_width_half+(PADDING_BRANK)), dtype=np.uint8)
    test_sample_mask[test_height_start_point[set]: test_height_end_point[set], test_width_start_point[set]: test_width_end_point[set], :] = np.ones(shape=(process_height_half+(PADDING_BRANK), process_width_half+(PADDING_BRANK), 3), dtype=np.uint8)
    test_answer_mask[test_height_start_point[set]: test_height_end_point[set], test_width_start_point[set]: test_width_end_point[set]] = np.ones(shape=(process_height_half+(PADDING_BRANK), process_width_half+(PADDING_BRANK)), dtype=np.uint8)

    learn_sample_image = sample_image * learn_sample_mask
    learn_answer_image = answer_image * learn_answer_mask
    test_sample_image = sample_image * test_sample_mask
    test_answer_image = answer_image * test_answer_mask

    total_learn_data = 0
    with tf.python_io.TFRecordWriter(out_learn_file_name) as learn_writer:
        with open(out_folder_name + '/' + 'learn_file_log' + '.txt', 'w') as learn_text_file:
            h_count = int((process_height - (PADDING_BRANK * 2)) / (HEIGHT_CROP_SIZE))
            w_count = int((process_width - (PADDING_BRANK * 2)) / (WIDTH_CROP_SIZE))
            pbar = tqdm(total=h_count, desc='learn Processed')  # プロセスバーの設定
            for h in range(h_count):
                crop_top = h * HEIGHT_CROP_SIZE
                crop_bottom = (h * HEIGHT_CROP_SIZE) + HEIGHT_CROP_SIZE + (PADDING_BRANK * 2)
                for w in range(w_count):
                    crop_left = w * WIDTH_CROP_SIZE
                    crop_right = (w * WIDTH_CROP_SIZE) + WIDTH_CROP_SIZE + (PADDING_BRANK * 2)

                    if set == 0:
                        if crop_top < test_height_end_point[set] and crop_left < test_width_end_point[set]:
                            continue
                    elif set == 1:
                        if crop_top < test_height_end_point[set] and crop_right >= test_width_start_point[set]:
                            continue
                    elif set == 2:
                        if crop_bottom >= test_height_start_point[set] and crop_left < test_width_end_point[set]:
                            continue
                    elif set == 3:
                        if crop_bottom >= test_height_start_point[set] and crop_right >= test_width_start_point[set]:
                            continue

                    sample_crop_data = learn_sample_image[crop_top: crop_bottom, crop_left: crop_right, :]
                    # sample_crop_rbg = cv2.cvtColor(sample_crop_data, cv2.COLOR_BGR2RGB)
                    sample_crop_data = sample_crop_data.astype(np.uint8)
                    answer_crop_data = learn_answer_image[crop_top: crop_bottom, crop_left: crop_right]
                    answer_crop_data = answer_crop_data
                    answer_crop_data = answer_crop_data.astype(np.uint8)

                    cood_Y = crop_top // HEIGHT_CROP_SIZE
                    cood_X = crop_left // WIDTH_CROP_SIZE

                    strY = str(cood_Y)
                    if cood_Y / 100 < 1:
                        strY = "0" + str(cood_Y)
                    if cood_Y / 10 < 1:
                        strY = "00" + str(cood_Y)

                    strX = str(cood_X)
                    if cood_X / 100 < 1:
                        strX = "0" + str(cood_X)
                    if cood_X / 10 < 1:
                        strX = "00" + str(cood_X)

                    sample_path = out_sample_folder + '/' + 'Sample' + '_' + 'X' + strX + '_' + 'Y' + strY + '.png'
                    answer_path = out_answer_folder + '/' + 'Answer' + '_' + 'X' + strX + '_' + 'Y' + strY + '.png'
                    img_dir_name = 'img2'

                    # if os.path.exists(sample_path) == False:
                    #     cv2.imwrite(sample_path, sample_crop_data)  # 判定画像出力
                    # if os.path.exists(answer_path) == False:
                    #     cv2.imwrite(answer_path, answer_crop_data)

                    example = tf.train.Example(features=tf.train.Features(feature={
                        'coordinate_Y': tf.train.Feature(int64_list=tf.train.Int64List(value=[cood_Y])),
                        'coordinate_X': tf.train.Feature(int64_list=tf.train.Int64List(value=[cood_X])),
                        'image_height': tf.train.Feature(int64_list=tf.train.Int64List(value=[HEIGHT_CROP_SIZE])),
                        'image_width': tf.train.Feature(int64_list=tf.train.Int64List(value=[WIDTH_CROP_SIZE])),
                        'padding_brank': tf.train.Feature(int64_list=tf.train.Int64List(value=[PADDING_BRANK])),
                        'image_name': tf.train.Feature(
                            bytes_list=tf.train.BytesList(value=[img_dir_name.encode()])),
                        'sample_path': tf.train.Feature(
                            bytes_list=tf.train.BytesList(value=[sample_path.encode()])),
                        'answer_path': tf.train.Feature(
                            bytes_list=tf.train.BytesList(value=[answer_path.encode()])),
                    }))
                    learn_writer.write(example.SerializeToString())
                    learn_text_file.writelines(
                        str(SAMPLE_FILE) + ': ' + str(total_learn_data) + ': ' + 'h:' + str(h) + ' w:' + str(w) + '\n')
                    total_learn_data = total_learn_data + 1
                pbar.update(1)  # プロセスバーを進行
            pbar.close()  # プロセスバーの終了



    total_test_data = 0
    with tf.python_io.TFRecordWriter(out_test_file_name) as test_writer:
        with open(out_folder_name + '/' + 'test_file_log' + '.txt', 'w') as test_text_file:
            h_list = list(range(test_height_start_point[set], test_height_end_point[set] - (PADDING_BRANK*2) - HEIGHT_CROP_SIZE + 1, HEIGHT_CROP_SIZE))
            pbar = tqdm(total=int(len(h_list)), desc='test Processed')  # プロセスバーの設定
            for h in range(len(h_list)):
                crop_top = h_list[h]
                crop_bottom = crop_top + HEIGHT_CROP_SIZE + (PADDING_BRANK * 2)
                w_list = list(range(test_width_start_point[set], test_width_end_point[set] - (PADDING_BRANK*2) - WIDTH_CROP_SIZE + 1, WIDTH_CROP_SIZE))
                for w in range(len(w_list)):
                    crop_left = w_list[w]
                    crop_right = crop_left + WIDTH_CROP_SIZE + (PADDING_BRANK * 2)

                    sample_crop_data = test_sample_image[crop_top: crop_bottom, crop_left: crop_right, :]
                    # sample_crop_rbg = cv2.cvtColor(sample_crop_data, cv2.COLOR_BGR2RGB)
                    sample_crop_data = sample_crop_data.astype(np.uint8)
                    answer_crop_data = test_answer_image[crop_top: crop_bottom, crop_left: crop_right]
                    answer_crop_data = answer_crop_data
                    answer_crop_data = answer_crop_data.astype(np.uint8)

                    cood_Y = crop_top // HEIGHT_CROP_SIZE
                    cood_X = crop_left // WIDTH_CROP_SIZE

                    strY = str(cood_Y)
                    if cood_Y / 100 < 1:
                        strY = "0" + str(cood_Y)
                    if cood_Y / 10 < 1:
                        strY = "00" + str(cood_Y)

                    strX = str(cood_X)
                    if cood_X / 100 < 1:
                        strX = "0" + str(cood_X)
                    if cood_X / 10 < 1:
                        strX = "00" + str(cood_X)

                    sample_path = out_sample_folder + '/' + 'Sample' + '_' + 'X' + strX + '_' + 'Y' + strY + '.png'
                    answer_path = out_answer_folder + '/' + 'Answer' + '_' + 'X' + strX + '_' + 'Y' + strY + '.png'
                    img_dir_name = 'img2'

                    # if os.path.exists(sample_path) == False:
                    #     cv2.imwrite(sample_path, sample_crop_data)  # 判定画像出力
                    # if os.path.exists(answer_path) == False:
                    #     cv2.imwrite(answer_path, answer_crop_data)

                    example = tf.train.Example(features=tf.train.Features(feature={
                        'coordinate_Y': tf.train.Feature(int64_list=tf.train.Int64List(value=[cood_Y])),
                        'coordinate_X': tf.train.Feature(int64_list=tf.train.Int64List(value=[cood_X])),
                        'image_height': tf.train.Feature(int64_list=tf.train.Int64List(value=[HEIGHT_CROP_SIZE])),
                        'image_width': tf.train.Feature(int64_list=tf.train.Int64List(value=[WIDTH_CROP_SIZE])),
                        'padding_brank': tf.train.Feature(int64_list=tf.train.Int64List(value=[PADDING_BRANK])),
                        'image_name': tf.train.Feature(
                            bytes_list=tf.train.BytesList(value=[img_dir_name.encode()])),
                        'sample_path': tf.train.Feature(
                            bytes_list=tf.train.BytesList(value=[sample_path.encode()])),
                        'answer_path': tf.train.Feature(
                            bytes_list=tf.train.BytesList(value=[answer_path.encode()])),
                    }))
                    test_writer.write(example.SerializeToString())
                    test_text_file.writelines(
                        str(SAMPLE_FILE) + ': ' + str(total_test_data) + ': ' + 'h:' + str(h) + ' w:' + str(w) + '\n')
                    total_test_data = total_test_data + 1
                pbar.update(1)  # プロセスバーを進行
            pbar.close()  # プロセスバーの終了

    # =====propertyデータの作成=======================================================

    LEARN_HEIGHT_CROP_SIZE = 480
    LEARN_WIDTH_CROP_SIZE = 640
    TEST_HEIGHT_CROP_SIZE = 480
    TEST_WIDTH_CROP_SIZE = 640
    with tf.python_io.TFRecordWriter(out_property_file_name) as writer:
        example = tf.train.Example(features=tf.train.Features(feature={
            'learn_image_height': tf.train.Feature(int64_list=tf.train.Int64List(value=[HEIGHT_CROP_SIZE])),
            'learn_image_width': tf.train.Feature(int64_list=tf.train.Int64List(value=[WIDTH_CROP_SIZE])),
            'test_image_height': tf.train.Feature(int64_list=tf.train.Int64List(value=[HEIGHT_CROP_SIZE])),
            'test_image_width': tf.train.Feature(int64_list=tf.train.Int64List(value=[WIDTH_CROP_SIZE])),
            'total_learn_data': tf.train.Feature(int64_list=tf.train.Int64List(value=[total_learn_data])),
            'total_test_data': tf.train.Feature(int64_list=tf.train.Int64List(value=[total_test_data])),
            'padding_brank': tf.train.Feature(int64_list=tf.train.Int64List(value=[PADDING_BRANK])),
            'data_channel': tf.train.Feature(int64_list=tf.train.Int64List(value=[3]))
        }))
        # レコード書込
        writer.write(example.SerializeToString())

    # recordファイルの内容をテキストファイルに記載
    with open(out_folder_name + '/' + OUTPUT_TEXT_FILE + '.txt', 'w') as text_file:
        # text_file = open( out_folder_name + '/' +OUTPUT_TEXT_FILE + '.txt', 'w')# 結果のテキストメモを生成
        text_file.writelines('learn_image_height: ' + str(HEIGHT_CROP_SIZE) + '\n')
        text_file.writelines('learn_image_width: ' + str(WIDTH_CROP_SIZE) + '\n')
        text_file.writelines('test_image_height: ' + str(HEIGHT_CROP_SIZE) + '\n')
        text_file.writelines('test_image_width: ' + str(WIDTH_CROP_SIZE) + '\n')
        text_file.writelines('total_learn_data: ' + str(total_learn_data) + '\n')
        text_file.writelines('total_test_data: ' + str(total_test_data) + '\n')
        text_file.writelines('data_channel: ' + str(3) + '\n')


    header = ['learn_image_height',
              'learn_image_width',
              'test_image_height',
              'test_image_width',
              'total_learn_data',
              'total_validation_data',
              'total_test_data',
              'padding_brank',
              'data_channel']
    val = [HEIGHT_CROP_SIZE,
           WIDTH_CROP_SIZE,
           HEIGHT_CROP_SIZE,
           WIDTH_CROP_SIZE,
           total_learn_data,
           total_test_data,
           PADDING_BRANK,
           3,]

    out_property_csv_name = out_folder_name +'/'+OUTPUT_PROPERTY_FILE+'-'+SETNAME[set]+'.csv'
    with open(out_property_csv_name, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerow(val)
