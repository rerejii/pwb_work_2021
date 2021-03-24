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

IN_PATH = 'Z:/hayakawa/work20/dataset2/'
OUT_PATH = 'Z:/hayakawa/work20/dataset2/mini_img/'

# out_folder_name = 'test_otsu_hls_s_2'
origin_folder = IN_PATH + 'img/*/*_brank_otsu_h.png'
ans_folder = IN_PATH + 'img/*/*_bin.png'
in_img_path = natsorted(glob.glob(origin_folder))
ans_img_path = natsorted(glob.glob(ans_folder))

with open('otsu_h_acc.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(['name', 'acc'])
    total_acc = 0
    for i in range(len(in_img_path)):
        print('Image %d / %d' % (i + 1, len(in_img_path)))
        in_img = cv2.imread(in_img_path[i], 0)
        ans_img = cv2.imread(ans_img_path[i], 0)
        h, w = in_img.shape
        kernel = np.ones((3, 3), np.uint8)
        closing = cv2.morphologyEx(in_img, cv2.MORPH_CLOSE, kernel)
        opening = cv2.morphologyEx(closing , cv2.MORPH_OPEN, kernel)
        acc = np.sum(opening == ans_img) / (h*w)
        print(acc)
        writer.writerow([in_img_path[i], acc])
        total_acc += acc
    total_acc = total_acc / len(in_img_path)
    print('total acc: ' + str(total_acc))
    writer.writerow(['total ave acc', total_acc])


# out_folder_name = OUT_PATH + out_folder_name
# os.makedirs(out_folder_name, exist_ok=True)