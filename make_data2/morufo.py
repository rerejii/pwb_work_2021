import cv2
import numpy as np
import glob
import os
import csv
from natsort import natsorted
import sys

kernel_size = 21



OUT_PATH = 'E:/work/myTensor/dataset2/'
# # OUT_PATH = '/home/hayakawa/work/myTensor/dataset2/'
# # OUT_PATH = 'C:/Users/hayakawa/work/mytensor/dataset2/'
origin_folder = OUT_PATH + 'img/*/*_bin.png'
opening_name = 'opening'
opening_sub_name = 'opening_sub'

in_img_path = [path for path in glob.glob(origin_folder)]
in_img_path = natsorted(in_img_path)

with open(OUT_PATH + 'img/morufo_rate.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(['binary_w_rate', 'binary_t_rate', 'nega_binary_w_rate', 'nega_binary_t_rate'])

    for i in range(len(in_img_path)):
        print(in_img_path[i])
        img = cv2.imread(in_img_path[i], 0)
        n_img = cv2.bitwise_not(img)
        h, w = img.shape
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
        opening_sub = np.abs(img - opening)
        nega_opening = cv2.morphologyEx(n_img, cv2.MORPH_OPEN, kernel)
        nega_opening_sub = np.abs(n_img - nega_opening)
        out_name = os.path.dirname(in_img_path[i]) + '/' + os.path.splitext(os.path.basename(in_img_path[i]))[0]
        morphology_weight = np.array(opening_sub + nega_opening_sub, dtype=np.uint8)

        file_name_list = []
        file_name_list.append(out_name + '_opening_' + str(kernel_size) + '.png')
        file_name_list.append(out_name + '_opening_sub_' + str(kernel_size) + '.png')
        file_name_list.append(out_name + '_nega_opening_' + str(kernel_size) + '.png')
        file_name_list.append(out_name + '_nega_opening_sub_' + str(kernel_size) + '.png')
        file_name_list.append(out_name + '_morphology-weight_' + str(kernel_size) + '.png')
        outimg_list = [opening, opening_sub, nega_opening, nega_opening_sub, morphology_weight]

        for i in range(len(outimg_list)):
            if os.path.isfile(file_name_list[i]):
                os.remove(file_name_list[i])
            cv2.imwrite(file_name_list[i], outimg_list[i])

        binary_w_n = np.sum(img / 255)
        binary_b_n = np.sum(n_img / 255)
        binary_t_n = h * w

        opening_sub_n = np.sum(opening_sub / 255)
        nega_opening_sub_n = np.sum(opening_sub / 255)

        binary_w_rate = opening_sub_n / binary_w_n
        binary_t_rate = opening_sub_n / binary_t_n
        nega_binary_b_rate = nega_opening_sub_n / binary_b_n
        nega_binary_t_rate = nega_opening_sub_n / binary_t_n

        writer.writerow([binary_w_rate, binary_t_rate, nega_binary_b_rate, nega_binary_t_rate])










