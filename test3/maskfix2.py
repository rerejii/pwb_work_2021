import cv2
import math
from natsort import natsorted
import glob
import os
import numpy as np
from tqdm import tqdm
import csv
import sys

# C:/Users/hayakawa/work/mytensor/dataset2
root_sam_path = 'C:/Users/hayakawa/work/mytensor/dataset2/img/17H-0863-1_L0012/L0012.png'
root_fix_path = 'Z:/hayakawa/binary/20210128/unet_use-bias_beta_otsu2/unet_use-bias_beta_otsu2-D/generator_L0012_fix.png'
root_ans_path = 'C:/Users/hayakawa/work/mytensor/dataset2/img/17H-0863-1_L0012/L0012_bin_oldfix.png'
root_out_path = 'C:/Users/hayakawa/work/mytensor/dataset2/img/17H-0863-1_L0012/L0012_maskline.png'
root_miniout_path = 'C:/Users/hayakawa/work/mytensor/dataset2/17H-0863-1_L0012/L0012_unet-maskline'
root_minians_path = 'C:/Users/hayakawa/work/mytensor/dataset2/17H-0863-1_L0012/L0012_unet-fixans'
root_out_ans_path = 'C:/Users/hayakawa/work/mytensor/dataset2/img/17H-0863-1_L0012/L0012_bin.png'

csv_path = 'maskfix.csv'
os.makedirs(root_miniout_path, exist_ok=True)
os.makedirs(root_minians_path, exist_ok=True)

CROP_H = 256
CROP_W = 256
PADDING = 20

def evenization(n):  # 偶数化
    return n + 1 if n % 2 != 0 else n

def img_size_norm(img):
    if img.ndim == 3:
        origin_h, origin_w, origin_ch = img.shape[:6]
    else:
        origin_h, origin_w = img.shape[:2]

    sheets_h = math.ceil(origin_h / CROP_H)  # math.ceil 切り上げ
    sheets_w = math.ceil(origin_w / CROP_W)  # math.ceil 切り上げ

    # 必要となる画像サイズを求める(外側切り出しを考慮してpadding*2を足す)
    flame_h = sheets_h * CROP_H + (PADDING * 2)
    flame_w = sheets_w * CROP_W + (PADDING * 2)

    # 追加すべき画素数を求める
    extra_h = flame_h - origin_h
    extra_w = flame_w - origin_w

    top, bottom = math.floor(extra_h / 2), flame_h - math.ceil(extra_h / 2)  # 追加画素数が奇数なら下右側に追加させるceil
    left, right = math.floor(extra_w / 2), flame_w - math.ceil(extra_w / 2)  # 追加画素数が奇数なら下右側に追加させるceil

    if img.ndim == 3:
        flame = np.zeros([flame_h, flame_w, 3], dtype=np.uint8)
        flame[top:bottom, left:right, :] = img
    else:
        flame = np.zeros([flame_h, flame_w], dtype=np.uint8)
        flame[top:bottom, left:right] = img

    return flame

def check_crop_index(norm_img):
    norm_h, norm_w = norm_img.shape[:2]
    h_count = int(norm_h / CROP_H)
    w_count = int(norm_w / CROP_W)
    crop_h = [n // w_count for n in range(h_count * w_count)]
    crop_w = [n % w_count for n in range(h_count * w_count)]
    crop_top = [n * CROP_H for n in crop_h]
    crop_left = [n * CROP_W for n in crop_w]
    crop_index = list(zip(*[crop_top, crop_left]))
    return crop_index

img_sam = cv2.imread(root_sam_path)
img_fix = cv2.imread(root_fix_path, 0)
img_ans = cv2.imread(root_ans_path, 0)
norm_sam = img_size_norm(img_sam)
norm_fix = img_size_norm(img_fix)
norm_ans = img_size_norm(img_ans)
crop_index = check_crop_index(img_fix)

norm_h, norm_w, _ = norm_sam.shape
norm_h = norm_h - (PADDING * 2)
norm_w = norm_w - (PADDING * 2)
out_flame = np.zeros(shape=[norm_h, norm_w], dtype=np.float32)

with tqdm(total=len(crop_index), desc='Processed') as pbar, open(csv_path, 'w') as f:
    writer = csv.writer(f)
    writer.writerow(['imgname', 'accuracy', 'move_h', 'move_w'])
    for ci in range(len(crop_index)):
        crop_top, crop_left = crop_index[ci]
        h = int(crop_top / CROP_H)
        w = int(crop_left / CROP_W)
        sam_crop = norm_sam[crop_top: crop_top + CROP_H + (PADDING * 2),
                       crop_left: crop_left + CROP_W + (PADDING * 2), :]
        fix_crop = norm_fix[crop_top: crop_top + CROP_H + (PADDING * 2),
                       crop_left: crop_left + CROP_W + (PADDING * 2)]
        ans_crop = norm_ans[crop_top: crop_top + CROP_H + (PADDING * 2),
                       crop_left: crop_left + CROP_W + (PADDING * 2)]
        str_Y = str(h).zfill(3)  # 左寄せゼロ埋め
        str_X = str(w).zfill(3)  # 左寄せゼロ埋め

        sam_mini = sam_crop[PADDING: -PADDING, PADDING: -PADDING]
        fix_mini = fix_crop[PADDING: -PADDING, PADDING: -PADDING]
        ans_mini = ans_crop[PADDING: -PADDING, PADDING: -PADDING]
        acc = np.average(fix_mini == ans_mini)

        max_acc = acc
        good_ans = ans_mini
        good_h = 0
        good_w = 0
        hw_list = [[math.floor(mi / (PADDING * 2 + 1)), mi % (PADDING * 2 + 1)] for mi in range((PADDING*2+1)*(PADDING*2+1))]
        if acc != 1.0:  # ずらす必要がないなら飛ばす
            for hw in hw_list:
                mh = hw[0]
                mw = hw[1]
                # fix_mini = fix_crop[mh: mh + CROP_H, mw: mw + CROP_W]
                ans_mini = ans_crop[mh: mh + CROP_H, mw: mw + CROP_W]
                acc = np.average(fix_mini == ans_mini)
                if acc > max_acc:
                    max_acc = acc
                    good_ans = ans_mini
                    good_h = mh - PADDING
                    good_w = mw - PADDING

        # out_flame[crop_top: crop_top + CROP_H, crop_left: crop_left + CROP_W] = good_ans
        # out_path = root_minians_path + '/Answer_X' + str_X + '_Y' + str_Y + '.png'
        # cv2.imwrite(out_path, good_ans)

        img_name = 'maskline_X' + str_X + '_Y' + str_Y + '.png'
        writer.writerow([img_name, max_acc, good_h, good_w])



        # 境界画像
        # mask = good_ans
        # inv_mask = cv2.bitwise_not(mask)
        # dst = cv2.distanceTransform(mask, cv2.DIST_L1, 5)
        # sam_mini[dst == 1] = [0, 255, 0]
        # inv_dst = cv2.distanceTransform(inv_mask, cv2.DIST_L1, 5)
        # sam_mini[inv_dst == 1] = [255, 0, 0]
        # out_path = root_miniout_path + '/maskline_X' + str_X + '_Y' + str_Y + '.png'
        # cv2.imwrite(out_path, sam_mini)

        pbar.update(1)  # プロセスバーを進行
# cv2.imwrite(root_out_ans_path, out_flame)


