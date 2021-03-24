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
OUT_PATH = 'Z:/hayakawa/work20/dataset2/mini_img/'

out_folder_name = 'test_otsu_hls_s_2'
origin_folder = IN_PATH + 'img/*/*_hls_s.png'
in_img_path = glob.glob(origin_folder)

out_folder_name = OUT_PATH + out_folder_name
os.makedirs(out_folder_name, exist_ok=True)
for pi in range(len(in_img_path)):
    print('Image %d / %d' % (pi + 1, len(in_img_path)))
    path = in_img_path[pi]
    img = cv2.imread(path, 0)
    height = img.shape[0]
    width = img.shape[1]
    ret2, img_otsu = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)
    img = img - ret2 + 127
    img2 = np.zeros([height, width])
    img2 += (img >= 127) * 255
    # img2 = cv2.resize(img_otsu, (int(width * 0.125), int(height * 0.125)))
    img2 = cv2.resize(img2, (int(width * 0.125), int(height * 0.125)))

    # cv2.imwrite(os.path.splitext(path)[0] + '_d25.png', img2)
    # cv2.imwrite(out_folder_name + '/' + os.path.splitext( os.path.basename(path) )[0] + '.png', img_otsu)
    cv2.imwrite(out_folder_name + '/' + os.path.splitext(os.path.basename(path))[0] + 'd125.png', img2)