'''
DeepOtsu用にのっぺりとした画像を生成するプログラム

round(np.sum(pass_img[:, :, 0], dtype=np.uint64) / count_n)
の部分ではnp.sumのデフォルト出力np.uint32では、桁数が足りずに画素値がとても小さくなるため、
np.uint64を採用、場合によってはさらに変数型を見直せ
'''

import numpy as np
import cv2
import sys
import glob
import os
from tqdm import tqdm
import shutil

origin_folder = 'E:/work/myTensor/dataset/img/*/*.png'
sample_paths = [path for path in glob.glob(origin_folder) if path.find('_bin') is -1]
answer_paths = [path for path in glob.glob(origin_folder) if path.find('_bin') is not -1]
OUT_FOLDER = 'E:/work/myTensor/dataset/deepOtsuImg/'
# OUT_FILE = 'L1.png'


def ave_color(mask, img):
    mask3 = mask.repeat(3).reshape(sample_img.shape)
    pass_img = np.array(img * mask3, dtype=np.uint8)
    count_n = np.sum(mask)
    R = round(np.sum(pass_img[:, :, 0], dtype=np.uint64) / count_n)
    G = round(np.sum(pass_img[:, :, 1], dtype=np.uint64) / count_n)
    B = round(np.sum(pass_img[:, :, 2], dtype=np.uint64) / count_n)
    color = [R, G, B]
    return np.array(mask3 * color)


if __name__ == '__main__':
    with tqdm(total=len(sample_paths), desc='Processed') as pbar:
        for pi in range(len(sample_paths)):
            # 画像の呼び出し、マスクの生成
            sample_img = cv2.imread(sample_paths[pi])
            sample_img = cv2.cvtColor(sample_img, cv2.COLOR_BGR2RGB)
            height, width, channel = sample_img.shape
            mask = np.array((cv2.imread(answer_paths[pi], 0) / 255) > 0.5)
            n_mask = np.array(mask==False)

            # 画像の生成
            result = np.zeros(sample_img.shape)
            result += ave_color(mask, sample_img)
            result += ave_color(n_mask, sample_img)

            # 保存の下準備
            result = cv2.cvtColor(np.array(result, dtype=np.uint8), cv2.COLOR_RGB2BGR)
            file_name = os.path.basename(sample_paths[pi])[:-4] + '_deepotsu.png'
            dir_name = os.path.basename(os.path.dirname(sample_paths[pi]))
            out_name = OUT_FOLDER + dir_name + '/' + file_name
            os.makedirs(OUT_FOLDER + dir_name, exist_ok=True)

            # 保存
            cv2.imwrite(out_name, result)
            shutil.copy(sample_paths[pi], OUT_FOLDER + dir_name)
            shutil.copy(answer_paths[pi], OUT_FOLDER + dir_name)

            pbar.update(1)


