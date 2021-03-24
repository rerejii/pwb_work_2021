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
matplotlib.use('Agg')

IN_PATH = 'Z:/hayakawa/work20/dataset2/'
OUT_PATH = 'Z:/hayakawa/work20/dataset2/mini_img/'
brank = [546+100, 347+100, 330+100, 319+100]

# out_folder_name = 'otsu_hls_s_2'
# origin_folder = IN_PATH + 'img/*/*_hls_s.png'
# in_img_path = glob.glob(origin_folder)

out_folder_name = 'otsu_outbrank_v'
origin_folder = IN_PATH + 'img/*/*.png'
in_img_path = [path for path in glob.glob(origin_folder) if len(os.path.basename(path)) is 9]

out_folder_name = OUT_PATH + out_folder_name
os.makedirs(out_folder_name, exist_ok=True)

def separation(gray, th):
    # クラス分け
    g0, g1 = gray[gray < th], gray[gray >= th]

    # 画素数
    w0, w1 = len(g0), len(g1)
    # 画素値分散
    s0_2, s1_2 = g0.var(), g1.var()
    # 画素値平均
    m0, m1 = g0.mean(), g1.mean()
    # 画素値合計
    p0, p1 = g0.sum(), g1.sum()

    # クラス内分散
    sw_2 = w0 * s0_2 + w1 * s1_2
    # クラス間分散
    sb_2 = ((w0 * w1) / ((w0 + w1) * (w0 + w1))) * ((m0 - m1) * (m0 - m1))
    # 分離度
    if (sb_2 != 0):
        X = sb_2 / sw_2
    else:
        X = 0
    return X

# with open('otsu_ret_brank.csv', 'w') as f:
#     writer = csv.writer(f)
#     writer.writerow(['name', 'ret_l', 'ret_s'])
for pi in range(len(in_img_path)):
    print('Image %d / %d' % (pi + 1, len(in_img_path)))
    path = in_img_path[pi]
    img = cv2.imread(path)

    height = img.shape[0]
    width = img.shape[1]
    img_hls = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    brankimg = img_hls[brank[0]:-1 * brank[1], brank[2]:-1 * brank[3]]
    ret2, img_otsu = cv2.threshold(brankimg[:, :, 2], 0, 255, cv2.THRESH_OTSU)
    out = np.zeros([height, width])
    out[img_hls[:, :, 2] >= ret2] = 255
    cv2.imwrite(os.path.splitext(path)[0] + '_brank_otsu_v.png', out)
    img_otsu = cv2.resize(out, (int(width * 0.125), int(height * 0.125)))
    cv2.imwrite(out_folder_name + '/' + os.path.splitext(os.path.basename(path))[0] + '_brank_otsu_v_d125.png', img_otsu)


    # img2 = cv2.resize(out, (int(width * 0.125), int(height * 0.125)))
    # cv2.imwrite(out_folder_name + '/' + os.path.splitext(os.path.basename(path))[0] + 'd125.png', img2)
        # cv2.imwrite(os.path.splitext(path)[0] + '_brank_otsu_h.png', out)
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


