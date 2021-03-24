import glob
import numpy as np
import os
import sys
from tqdm import tqdm
import cv2

black = [0, 0, 0]  # 予測:0黒 解答:0黒
white = [255, 255, 255]  # 予測:1白 解答:1白
red = [255, 0, 0]  # 予測:1白 解答:0黒
blue = [0, 204, 255]  # 予測:0黒 解答:1白

out_folder = 'crop_img/distans_out'
os.makedirs(out_folder, exist_ok=True)
img_path_root = 'Z:/hayakawa/share/'
img_path = [
    'img/17H-0863-1_L0006-new/L0006_bin.png',
    'img/17H-0863-1_L0011/L0011_bin.png',
    'img/17H-0863-1_L0017/L0017_bin.png',
    'img/17H-0863-1_L0003/L0003_bin.png',
    'img/17H-0863-1_L0005/L0005_bin.png',
    'img/17H-0863-1_L0009/L0009_bin.png',
    'img/17H-0863-1_L0002-old/L0002_bin.png',
    'img/17H-0863-1_L0008/L0008_bin.png',
    'img/17H-0863-1_L0013/L0013_bin.png',
    'img/17H-0863-1_L0007/L0007_bin.png',
    'img/17H-0863-1_L0012/L0012_bin.png',
    'img/17H-0863-1_L0014/L0014_bin.png',
    'img/17H-0863-1_L0004/L0004_bin.png',
    'img/17H-0863-1_L0015/L0015_bin.png',
    'img/17H-0863-1_L0016/L0016_bin.png',
    'img/17H-0863-1_L0001/L0001_bin.png',
    'img/17H-0863-1_L0010/L0010_bin.png',
    'img/17H-0863-1_L0018/L0018_bin.png',
]

# def task(crop):
#     middle = crop[1, 1]
#     if middle <= 2:
#


def distans_weight(in_img):
    img = in_img
    inv = cv2.bitwise_not(img)
    dst = cv2.distanceTransform(img, cv2.DIST_L2, 5)
    inv_dst = cv2.distanceTransform(inv, cv2.DIST_L2, 5)
    # kernel = np.ones((3, 3), np.uint8)
    kernel = [[0,1,0,],
              [1,1,1,],
              [0,1,0,],]
    kernel = np.array(kernel, np.uint8)
    weight = np.zeros([h, w])
    dist_list = []
    for i in range(2):
        src = np.array(((dst - (i + 1)) >= 0), np.uint8)
        check = cv2.dilate(src, kernel, iterations=1)  # Trueの位置ならweightをおいても潰れない...はず
        weight += check * (dst == i)
    return weight
    # check_2 = cv2.dilate(((dst - 2) >= 0), kernel, iterations=1)  # Trueの位置ならweightをおいても潰れない
    # print(tmp)



# img_path_set = [img_path_root + path for path in img_path]
img_path_set = glob.glob('crop_img_2/*.png')
pbar = tqdm(total=len(img_path_set))
for img_idx in range(len(img_path_set)):
    img = cv2.imread(img_path_set[img_idx], 0)
    img_name = os.path.splitext(os.path.basename(img_path_set[img_idx]))[0]
    h, w = img.shape[:2]
    weight = distans_weight(img)
    result_img = np.zeros([h, w, 3])
    img_3d = img.repeat(3).reshape(h, w, 3) / 255
    w_3d = weight.repeat(3).reshape(h, w, 3)
    result_img += (img_3d == 1) * (w_3d == 0) * white
    result_img += (w_3d == 1) * red
    result_img = np.array(result_img, np.uint8)
    result_img = cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(out_folder + '/' + img_name + '_' + '.png', result_img)
    pbar.update(1)  # プロセスバーを進行
pbar.close()  # プロセスバーの終了





