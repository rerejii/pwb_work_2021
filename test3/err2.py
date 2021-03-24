import cv2
import numpy as np
import glob
import os

# # 評価画像の出力
black = [0, 0, 0]  # 予測:0黒 解答:0黒
white = [255, 255, 255]  # 予測:1白 解答:1白
# red = [255, 0, 0]  # 予測:1白 解答:0黒
# blue = [0, 204, 255]  # 予測:0黒 解答:1白
red = [0, 0, 255]
blue = [255, 204, 0]

path = 'Z:/hayakawa/work20/tensor/test/crop_img/L4/*.png'
mask_path = 'Z:/hayakawa/work20/tensor/test/crop_img/L4/sam_4_bin.png'
out_path = 'Z:/hayakawa/work20/tensor/test/crop_img/L4/out2/'

img_pathset = glob.glob(path)
mask = cv2.imread(mask_path, 0)
height, width = mask.shape[:2]
mask3 = mask.repeat(3).reshape(height, width, 3) / 255
for i in range(len(img_pathset)):
    print(img_pathset[i])
    img = cv2.imread(img_pathset[i])

    # result_img = np.zeros([height, width, 3])
    # img3 = img.repeat(3).reshape(height, width, 3) / 255
    # result_img += (img3 == 0) * (mask3 == 0) * black  # 黒 正解
    # result_img += (img3 == 1) * (mask3 == 1) * white  # 黒 正解
    # result_img += (img3 == 1) * (mask3 == 0) * red # 黒 正解
    # result_img += (img3 == 0) * (mask3 == 1) * blue  # 黒 正解
    cv2.imwrite(out_path + os.path.basename(img_pathset[i]), img[256:-256, 512:])
