import cv2
import numpy as np
import matplotlib
import os
from matplotlib import pyplot as plt
from tqdm import tqdm
from natsort import natsorted
import pandas as pd
import glob
matplotlib.use('Agg')

IN_PATH = 'Z:/hayakawa/work20/dataset2/'
origin_folder = IN_PATH + 'img/*/*.png'
in_img_path = [path for path in glob.glob(origin_folder) if len(os.path.basename(path)) is 9]

for pi in range(len(in_img_path)):
    print('Image %d / %d' % (pi + 1, len(in_img_path)))
    path = in_img_path[pi]
    img = cv2.imread(path)
    img_hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    ret2_l, img_otsu_l = cv2.threshold(img_hls[:, :, 1], 0, 255, cv2.THRESH_OTSU)
    ret2_s, img_otsu_s = cv2.threshold(img_hls[:, :, 2], 0, 255, cv2.THRESH_OTSU)
    # out_l = 127 + img_hls[:, :, 1] - ret2_l
    # out_s = 127 + img_hls[:, :, 2] - ret2_s
    out_l = img_hls[:, :, 1] - ret2_l + 127
    out_s = img_hls[:, :, 2] - ret2_s + 127
    cv2.imwrite(os.path.splitext(path)[0] + '_otsu127_l.png', out_l)
    cv2.imwrite(os.path.splitext(path)[0] + '_otsu127_s.png', out_s)
    # out = np.zeros(img_hls.shape)
    # out[:, :, 1:] += img_hls[:, :, 1:]
    # out[:, :, 0] += img_hls[:, :, 1]
    # out[:, :, 2] += img_hls[:, :, 2]
    # cv2.imwrite(os.path.splitext(path)[0] + '_ls2.png', out)
    # cv2.imwrite(os.path.splitext(path)[0] + '_hls_l.png', img_hls[:, :, 1])
    # cv2.imwrite(os.path.splitext(path)[0] + '_hls_s.png', img_hls[:, :, 2])
    # cv2.imwrite(os.path.splitext(path)[0] + '_hls_h.png', img_hls[:, :, 0])
    # img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # cv2.imwrite(os.path.splitext(path)[0] + '_gray.png', img_gray)



# img = cv2.imread('Sample_X012_Y031.png')
# print(img)
# img_hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
# print(img_hls)
# cv2.imwrite('Sample_X012_Y031_hls.png', img_hls)
# os.path.splitext