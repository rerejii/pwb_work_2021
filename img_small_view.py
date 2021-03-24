import cv2
import numpy as np
import os
import glob
import sys
from natsort import natsorted

TARGET_NAME = 'unet_use-bias_beta_otsu2'

IN_PATH = 'Z:/hayakawa/binary/20210128/' + TARGET_NAME
OUT_PATH = 'Z:/hayakawa/work20/dataset2/mini_img/' + TARGET_NAME + '/'
origin_folder = IN_PATH + '/*/View_L00??.png'
in_img_path = [path for path in glob.glob(origin_folder)]
in_img_path = natsorted(in_img_path)
os.makedirs(OUT_PATH, exist_ok=True)

for pi in range(len(in_img_path)):
    print('Image %d / %d' % (pi + 1, len(in_img_path)))
    path = in_img_path[pi]
    img = cv2.imread(path)
    height = img.shape[0]
    width = img.shape[1]
    img2 = cv2.resize(img, (int(width * 0.5), int(height * 0.5)))
    cv2.imwrite(OUT_PATH + os.path.splitext(os.path.basename(path))[0] + '_d5.png', img2)


# img = cv2.imread(path)
# height = img.shape[0]
# width = img.shape[1]
# img2 = cv2.resize(img, (int(width * 0.5), int(height * 0.5)))
# cv2.imwrite(os.path.splitext(path)[0] + 'd5.png', img2)
