import os
import numpy as np
import cv2
import sys
import math
import random
from tqdm import tqdm
from check_brank import check_brank


CROP_H = 256
CROP_W = 256
PADDING = 28
RANDOM_SEED = 1
SAMPLE_FILE = 'E:/work/myTensor/dataset/img/17H-0863-1_L0002-new/L0002.png'
ANSWER_FILE = 'E:/work/myTensor/dataset/img/17H-0863-1_L0002-new/L0002_bin.png'
OUT_FOLDER = 'E:/work/myTensor/dataset2/Img-256-valid'+str(PADDING)+'-L2-img'

sam_folder = OUT_FOLDER+'/'+'Sample'
ans_folder = OUT_FOLDER+'/'+'Answer'
os.makedirs(sam_folder, exist_ok=True)
os.makedirs(ans_folder, exist_ok=True)


def evenization(n):  # 偶数化
    return n + 1 if n % 2 != 0 else n

# 乱数設定
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# 画像の読み込み サイズ取得
indata = cv2.imread(SAMPLE_FILE)
img = np.array(indata, dtype=np.uint8)
origin_h, origin_w, _ = img.shape

# 画像のカット(周囲の黒除去)
top_brank, bottom_brank, left_brank, right_brank = check_brank(img)
sam_img = img[top_brank:origin_h - bottom_brank, left_brank:origin_w - right_brank, :]
in_ans = cv2.imread(ANSWER_FILE, 0)[top_brank:origin_h - bottom_brank, left_brank:origin_w - right_brank]
ans_img = np.array(in_ans, dtype=np.uint8)

# 切り取る枚数を計算(切り取り枚数は偶数とする)
c_img_h, c_img_w, _ = sam_img.shape
sheets_h = evenization(math.ceil(c_img_h / (CROP_H)))  # math.ceil 切り上げ
sheets_w = evenization(math.ceil(c_img_w / (CROP_W)))  # math.ceil 切り上げ

# 必要となる画像サイズを求める(外側切り出しを考慮してpadding*2を足す)
flame_h = sheets_h * CROP_H + (PADDING * 2)  # 偶数 * N * 偶数 = 偶数
flame_w = sheets_w * CROP_W + (PADDING * 2)

# 追加すべき画素数を求める
extra_h = flame_h - c_img_h  # if 偶数 - 奇数 = 奇数
extra_w = flame_w - c_img_w  # elif 偶数 - 偶数 = 偶数

# 必要画素数のフレームを作って中心に画像を挿入
sam_flame = np.zeros([flame_h, flame_w, 3], dtype=np.uint8)
ans_flame = np.zeros([flame_h, flame_w], dtype=np.uint8)
top, bottom = math.floor(extra_h / 2), flame_h - math.ceil(extra_h / 2)  # 追加画素が奇数なら下右側に追加させるceil
left, right = math.floor(extra_w / 2), flame_w - math.ceil(extra_w / 2)  # 追加画素が奇数なら下右側に追加させるceil
sam_flame[top:bottom, left:right, :] = sam_img
ans_flame[top:bottom, left:right] = ans_img

# 切り取りサイズ算出
crop_size_h = CROP_H + (PADDING * 2)
crop_size_w = CROP_W + (PADDING * 2)

# 開始位置と終了地点[A, B, C, D]
# 偶数は2で割ると少数が出る 左側(上側)は中央から右に行きたくないからfloor 右側(下側)は中央から左に行きたくないからceil
# startは左側はゼロやで考慮しなくていい、右側スタートを考慮してstartは全部ceilでいいはず(中央を超えた画素がスタート)
test_start_h = [0, 0, math.ceil(flame_h/2), math.ceil(flame_h/2)]
test_start_w = [0, math.ceil(flame_w/2), 0, math.ceil(flame_w/2)]
# endは右側は画像端に当たるから考慮なし、中央がendになる右側スタートだけ考慮する、から全部floorになるはず(中央を超える前にend)
test_end_h = [math.floor(flame_h/2), math.floor(flame_h/2), flame_w, flame_w]
test_end_w = [math.floor(flame_w/2), flame_w, math.floor(flame_w/2), flame_w]

# デバック用パラメータ
learn_count_li, test_count_li, remove_count_li = [], [], []
learn_count, test_count, remove_count = 0, 0, 0

# セット単位の実行
for set in range(4):
    h_count = int((flame_h - (PADDING * 2)) / (CROP_H))  # 切り出せる画像を求める
    w_count = int((flame_w - (PADDING * 2)) / (CROP_W))  # 切り出せる画像を求める
    with tqdm(total=h_count, desc=str(set) + ' set Processed') as pbar:
        for h in range(h_count):
            # 切り出しを行う座標を求める 切り出しの開始点+切り出し幅+パディング両側分
            point_top = h * CROP_H + PADDING # PADDINGを考慮しない時の切り出し頂点を算出
            point_bottom = point_top + CROP_H
            for w in range(w_count):
                point_left = w * CROP_W + PADDING # PADDINGを考慮しない時の切り出し頂点を算出
                point_right = point_left + CROP_W
                # testの範囲であるか否かの判定 start以上 and end以下
                # chect_testがtrueなら4隅すべてがテスト領域に存在することを示す
                check_test = (point_top >= test_start_h[set]) and \
                             (point_bottom <= test_end_h[set]) and \
                             (point_left >= test_start_w[set]) and \
                             (point_right <= test_end_w[set])
                # check_rmがtrueなら4隅全てが(テスト領域+1画像分)に踏み入れている(テスト領域に踏み込んでいる)
                check_rm = (point_top >= test_start_h[set] - CROP_H) and \
                           (point_bottom <= test_end_h[set] + CROP_H) and \
                           (point_left >= test_start_w[set] - CROP_W) and \
                           (point_right <= test_end_w[set] + CROP_W)
                if not check_test and check_rm:
                    remove_count += 1
                    continue
                # 切り取り座標の算出及び切り取り
                crop_top, crop_bottom = point_top - PADDING, point_bottom + PADDING
                crop_left, crop_right = point_left - PADDING, point_right + PADDING
                sam_crop = sam_flame[crop_top: crop_bottom, crop_left: crop_right, :]
                ans_crop = ans_flame[crop_top: crop_bottom, crop_left: crop_right]
                str_X = str(h).zfill(3)  # 左寄せゼロ埋め
                str_Y = str(w).zfill(3)  # 左寄せゼロ埋め
                sam_out_path = sam_folder + '/Sample_X' + str_X + '_Y' + str_Y + '.png'
                ans_out_path = ans_folder + '/Answer_X' + str_X + '_Y' + str_Y + '.png'
                cv2.imwrite(sam_out_path, sam_crop)
                cv2.imwrite(ans_out_path, ans_crop)
                test_count += check_test
                learn_count += not check_test
            pbar.update(1)  # プロセスバーを進行
        learn_count_li.append(learn_count)
        test_count_li.append(test_count)
        remove_count_li.append(remove_count)
        learn_count, test_count, remove_count = 0, 0, 0


# デバック出力
setname = ['A', 'B', 'C', 'D']
for set in range(4):
    print('set' + setname[set] + ' learn_count: ' + str(learn_count_li[set]))
    print('set' + setname[set] + ' test_count: ' + str(test_count_li[set]))
    print('set' + setname[set] + ' remove_count: ' + str(remove_count_li[set]))






