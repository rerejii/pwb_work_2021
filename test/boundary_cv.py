import glob
import numpy as np
import os
import sys
from tqdm import tqdm
import cv2

boundary_kernel = 3
crush_kernel = 3
black = [0, 0, 0]  # 予測:0黒 解答:0黒
white = [255, 255, 255]  # 予測:1白 解答:1白
red = [255, 0, 0]  # 予測:1白 解答:0黒
blue = [0, 204, 255]  # 予測:0黒 解答:1白

dil_size = 30
out_folder = 'crop_img/boundary' + str(dil_size)
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
img_path = [img_path_root + path for path in img_path]

def binary_to_img(img):
    return np.greater_equal(img, 127)


def boundary5(in_img):
    kernel = in_img
    kernel = np.array([[0,1,1,1,0],
                      [1,1,1,1,1],
                      [1,1,1,1,1],
                      [1,1,1,1,1],
                      [0,1,1,1,0],], np.uint8)
    ero_img = cv2.erode(img, kernel, iterations=1)
    dil_img = cv2.dilate(img, kernel, iterations=1)
    weight = np.array(img != ero_img, np.uint8) + np.array(img != dil_img, np.uint8)
    weight = np.array(weight >= 1, np.uint8)
    return weight


def boundary(in_img, kernel_size=5):
    img = in_img
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    ero_img = cv2.erode(img, kernel, iterations=1)
    dil_img = cv2.dilate(img, kernel, iterations=1)
    weight = np.array(img != ero_img, np.uint8) + np.array(img != dil_img, np.uint8)
    weight = np.array(weight >= 1, np.uint8)
    return weight

def check_crush(in_img, kernel_size=5):
    img = in_img
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    weight = np.array(img != opening, np.uint8) + np.array(img != closing, np.uint8)
    weight = np.array(weight >= 1, np.uint8)
    return weight

def distans_weight(in_img, size):
    img = in_img
    inv = cv2.bitwise_not(img)
    dst = cv2.distanceTransform(img, cv2.DIST_L2, 5)
    inv_dst = cv2.distanceTransform(inv, cv2.DIST_L2, 5)
    dst[dst == 0] = size + 1
    inv_dst[inv_dst == 0] = size + 1
    weight = np.array(dst <= size, np.uint8) + np.array(inv_dst <= size, np.uint8)
    weight = np.array(weight >= 1, np.uint8)
    return weight

# def check_crush(in_img, kernel_size=5):
#     img = in_img
#     kernel = np.ones((3, 3), np.uint8)
#     ero_img = cv2.erode(img, kernel, iterations=20)
#     closing = cv2.dilate(ero_img, kernel, iterations=20)
#     dil_img = cv2.dilate(img, kernel, iterations=20)
#     opening = cv2.dilate(dil_img, kernel, iterations=20)
#     weight = np.array(img != opening, np.uint8) + np.array(img != closing, np.uint8)
#     weight = np.array(weight >= 1, np.uint8)
#     return weight


ans_binary_list = img_path
# ans_binary_list = glob.glob('crop_img_2/*.png')
pbar = tqdm(total=len(ans_binary_list))
for img_idx in range(len(ans_binary_list)):
    img = cv2.imread(ans_binary_list[img_idx],0)
    img_name = os.path.splitext(os.path.basename(ans_binary_list[img_idx]))[0]
    h, w = img.shape[:2]
    weight_3 = boundary(img, kernel_size=3)
    # weight_5 = boundary(img, kernel_size=5)
    weight_5 = boundary5(img)
    # crush_5 = check_crush(img, kernel_size=5)
    crush_7 = check_crush(img, kernel_size=7)
    # crush_big = check_crush(img, kernel_size=20)
    weight_d = distans_weight(img, size=dil_size)

    result_img = np.zeros([h, w, 3])
    img_3d = img.repeat(3).reshape(h, w, 3) / 255
    w3_3d = weight_3.repeat(3).reshape(h, w, 3)
    w5_3d = weight_5.repeat(3).reshape(h, w, 3)
    c7_3d = crush_7.repeat(3).reshape(h, w, 3)
    # c_big_3d = crush_big.repeat(3).reshape(h, w, 3)
    d_3d = weight_d.repeat(3).reshape(h, w, 3)
    # result_img += (w_3d == 0) * (img_3d == 0) * black
    boundary_w = np.zeros([h, w, 3])
    boundary_w += (w3_3d == 1)
    boundary_w += (w3_3d == 0) * (w5_3d == 1) * (c7_3d == 0)
    boundary_w = np.array(boundary_w >= 1, np.uint8)
    dil_w = np.zeros([h, w, 3])
    dil_w += (boundary_w == 0) * (d_3d == 1)
    dil_w = np.array(dil_w >= 1, np.uint8)
    # print(crush_w)

    # result_img += white * (boundary_w == 0) * (img_3d == 1)
    result_img += white * (dil_w == 0) * (img_3d == 1)
    result_img += red * (boundary_w == 1)
    result_img += blue * (dil_w == 1)
    # result_img += blue * (boundary_w == 0) * (crush_w == 0)



    # result_img += (img_3d == 1) * white * (w3_3d == 0) * (w5_3d == 1) * (c7_3d == 1) * (c_big_3d == 0)
    # result_img += (img_3d == 1) * white * (w3_3d == 0) * (w5_3d == 0) * (c_big_3d == 0)
    # result_img += (w3_3d == 1) * red
    # result_img += (w3_3d == 0) * (w5_3d == 1) * (c7_3d == 0) * red
    # result_img += blue * (w3_3d == 0) * (w5_3d == 1) * (c7_3d == 1) * (c_big_3d == 0)
    # result_img += blue * (w3_3d == 0) * (w5_3d == 0) * (c_big_3d == 0)
    result_img = np.array(result_img, np.uint8)
    result_img = cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(out_folder + '/' + img_name + '_boundary' + str(dil_size) + '.png', result_img)

    boundary_w = np.zeros([h, w])
    boundary_w += (weight_3 == 1)
    boundary_w += (weight_3 == 0) * (weight_5 == 1) * (crush_7 == 0)
    boundary_w = np.array(boundary_w >= 1, np.uint8)
    cv2.imwrite(out_folder + '/' + img_name + '_boundary.png', boundary_w*255)

    # result_img = np.zeros([h, w, 3])
    # result_img += (w_3d == 0) * (img_3d == 0) * black
    # result_img += (w_3d == 0) * (img_3d == 1) * white
    # result_img += (c_3d == 1) * (img_3d == 0) * black
    # result_img += (c_3d == 1) * (img_3d == 1) * white
    # result_img += (w_3d == 1) * (c_3d == 0) * red
    # result_img = np.array(result_img, np.uint8)
    # result_img = cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR)
    # cv2.imwrite(out_folder + '/' + img_name + '_bk' + str(boundary_kernel) + '_ck' + str(crush_kernel) + '.png', result_img)
    pbar.update(1)  # プロセスバーを進行
pbar.close()  # プロセスバーの終了
