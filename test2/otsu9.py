import cv2
import numpy as np
import matplotlib
import os
from matplotlib import pyplot as plt
from tqdm import tqdm
from natsort import natsorted
import pandas as pd
import glob
import csv
import time as timefunc
import math
matplotlib.use('Agg')

IN_PATH = 'Z:/hayakawa/work20/dataset2/'
OUT_PATH = 'Z:/hayakawa/work20/dataset2/mini_img/'
brank = [546+100, 347+100, 330+100, 319+100]

# out_folder_name = 'otsu_hls_s_2'
# origin_folder = IN_PATH + 'img/*/*_hls_s.png'
# in_img_path = glob.glob(origin_folder)

out_folder_name = OUT_PATH + 'otsu_outbrank_lh75-25'
origin_folder = IN_PATH + 'img/*/*.png'
in_img_path = [path for path in glob.glob(origin_folder) if len(os.path.basename(path)) is 9]
os.makedirs(out_folder_name, exist_ok=True)


def get_str_time(time_num):
    day_time = 86400
    ms_time, s_time = math.modf(time_num)  # ミニセカンド セカンド
    day, times = divmod(s_time, day_time)  # 日数と時間に
    day = int(day)
    step_times = timefunc.strptime(timefunc.ctime(times))
    str_time = str(day) + 'd' + timefunc.strftime("%H:%M:%S", step_times) + str(ms_time)[1:]
    return str_time


c = 7
with open('otsu_ret_brank_c'+str(c)+'.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(['name', 'ret_l', 'time'])
    for pi in range(len(in_img_path)):
        print('Image %d / %d' % (pi + 1, len(in_img_path)))
        path = in_img_path[pi]
        img = cv2.imread(path)

        start = timefunc.time()

        height = img.shape[0]
        width = img.shape[1]
        img_hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)

        brankimg = img_hls[brank[0]:-1 * brank[1], brank[2]:-1 * brank[3], 1]
        ret2_l, img_otsu_l = cv2.threshold(brankimg, 0, 255, cv2.THRESH_OTSU)
        out_l = np.zeros([height, width])
        out_l[img_hls[:, :, 1] >= ret2_l] = 255

        kernel = np.ones((c, c), np.uint8)
        closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

        end = timefunc.time()

        time_str = get_str_time(end-start)

        writer.writerow([path, ret2_l, time_str])

        # cv2.imwrite(os.path.splitext(path)[0] + '_otsu_outbrank_lh75-25.png', out_l)
        # img2 = cv2.resize(out_l, (int(width * 0.125), int(height * 0.125)))
        # cv2.imwrite(out_folder_name + '/' + os.path.splitext(os.path.basename(path))[0] + '_otsu_outbrank_lh75-25_b1' + '_d125.png', img2)

        # for c in [3,5,7,9,11]:
        #     out_folder_name = 'otsu_outbrank_l_c'+str(c)
        #     out_folder_name = OUT_PATH + out_folder_name
        #     os.makedirs(out_folder_name, exist_ok=True)
        #     kernel = np.ones((c, c), np.uint8)
        #     closing = cv2.morphologyEx(out_l, cv2.MORPH_CLOSE, kernel)
        #     cv2.imwrite(os.path.splitext(path)[0] + '_brank_otsu_l_c'+str(c)+'.png', closing)
        #     img2 = cv2.resize(closing, (int(width * 0.125), int(height * 0.125)))
        #     cv2.imwrite(out_folder_name + '/' + os.path.splitext(os.path.basename(path))[0] + '_brank_otsu_l_c' + str(c) + '_d125.png', img2)

        # cv2.imwrite(out_folder_name + '/' + os.path.splitext(os.path.basename(path))[0] + 'l_d125.png', img_otsu_l)
        # cv2.imwrite(out_folder_name + '/' + os.path.splitext(os.path.basename(path))[0] + 's_d125.png', img_otsu_s)
        # writer.writerow([path, ret2_l, ret2_s])
        # dil_l = np.abs(np.sum(img_otsu_l == 0) - np.sum(img_otsu_l == 255))
        # dil_s = np.abs(np.sum(img_otsu_s == 0) - np.sum(img_otsu_s == 255))
        # print(dil_l)
        # print(dil_s)
        # # separation_l = separation(img_hls[:, :, 1], ret2_l)
        # # separation_h = separation(img_hls[:, :, 2], ret2_s)
        # if dil_l < dil_s:  # 差分が小さい方を選択
        #     cv2.imwrite(out_folder_name + '/' + os.path.splitext(os.path.basename(path))[0] + 'd125.png', img_otsu_l)
        #     writer.writerow([path, 'l', dil_l, dil_s])
        #     print('l')
        # else:
        #     cv2.imwrite(out_folder_name + '/' + os.path.splitext(os.path.basename(path))[0] + 'd125.png', img_otsu_s)
        #     writer.writerow([path, 's', dil_l, dil_s])
        #     print('s')

    # out_l = 127 + img_hls[:, :, 1] - ret2_l
    # out_h = 127 + img_hls[:, :, 2] - ret2_h
    # # cv2.imwrite(os.path.splitext(path)[0] + '_otsu127_select.png', out_l)
    # ret2, img_otsu = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)
    # img2 = img - ret2 + 127
    # # img2 = cv2.resize(img_otsu, (int(width * 0.125), int(height * 0.125)))
    # img2 = cv2.resize(img2, (int(width * 0.125), int(height * 0.125)))
    #
    # # cv2.imwrite(os.path.splitext(path)[0] + '_d25.png', img2)
    # # cv2.imwrite(out_folder_name + '/' + os.path.splitext( os.path.basename(path) )[0] + '.png', img_otsu)
    # cv2.imwrite(out_folder_name + '/' + os.path.splitext(os.path.basename(path))[0] + 'd125.png', img2)


