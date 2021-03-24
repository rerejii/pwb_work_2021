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

# out_folder_name = 'sample'
# origin_folder = IN_PATH + 'img/*/*.png'
# in_img_path = [path for path in glob.glob(origin_folder) if len(os.path.basename(path)) is 9]

# out_folder_name = 'gray'
# origin_folder = IN_PATH + 'img/*/*_gray.png'
# in_img_path = glob.glob(origin_folder)

# out_folder_name = 'hls_s'
# origin_folder = IN_PATH + 'img/*/*_hls_s.png'
# in_img_path = glob.glob(origin_folder)

# out_folder_name = 'hls_l'
# origin_folder = IN_PATH + 'img/*/*_hls_l.png'
# in_img_path = glob.glob(origin_folder)

# out_folder_name = 'hls_h'
# origin_folder = IN_PATH + 'img/*/*_hls_h.png'
# in_img_path = glob.glob(origin_folder)

out_folder_name = 'hls_ls'
origin_folder = IN_PATH + 'img/*/*_ls.png'
in_img_path = glob.glob(origin_folder)


out_folder_name = OUT_PATH + out_folder_name
os.makedirs(out_folder_name, exist_ok=True)
for pi in range(len(in_img_path)):
    print('Image %d / %d' % (pi + 1, len(in_img_path)))
    path = in_img_path[pi]
    img = cv2.imread(path)
    height = img.shape[0]
    width = img.shape[1]
    img2 = cv2.resize(img, (int(width * 0.125), int(height * 0.125)))

    # cv2.imwrite(os.path.splitext(path)[0] + '_d25.png', img2)
    cv2.imwrite(out_folder_name + '/' + os.path.splitext( os.path.basename(path) )[0] + '_d125.png', img2)