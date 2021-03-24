import cv2
import math
from natsort import natsorted
import glob
import os
# C:\Users\hayakawa\work\mytensor\dataset2
root_img_path = 'C:/Users/hayakawa/work/mytensor/dataset2/17H-0863-1_L0012/L0012'
root_ans_path = 'C:/Users/hayakawa/work/mytensor/dataset2/17H-0863-1_L0012/L0012_bin'
root_out_path = 'C:/Users/hayakawa/work/mytensor/dataset2/17H-0863-1_L0012/L0012_maskline'
os.makedirs(root_out_path, exist_ok=True)
img_path_set = glob.glob(root_img_path + '/*.png')
img_path_set = natsorted(img_path_set)
ans_path_set = glob.glob(root_ans_path + '/*.png')
ans_path_set = natsorted(ans_path_set)

for i in range(len(img_path_set)):
# for i in range(10):

    img_path = img_path_set[i]
    ans_path = ans_path_set[i]
    out_path = root_out_path + '/L0012_maskline_' + os.path.basename(img_path)[6:]
    img = cv2.imread(img_path)
    mask = cv2.imread(ans_path, 0)
    inv_mask = cv2.bitwise_not(mask)

    dst = cv2.distanceTransform(mask, cv2.DIST_L1, 5)
    img[dst==1] = [0, 255, 0]

    inv_dst = cv2.distanceTransform(inv_mask, cv2.DIST_L1, 5)
    img[inv_dst==1] = [255, 0, 0]

    cv2.imwrite(out_path, img)
