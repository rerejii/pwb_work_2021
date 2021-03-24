import numpy as np
import tensorflow as tf
from tqdm import tqdm
import sys
import csv



# A
# TEST: 3,5,9
# VALIDATION: 6,11,17
# A: ['img/17H-0863-1_L0006-new/L0006.png', 'img/17H-0863-1_L0011/L0011.png', 'img/17H-0863-1_L0017/L0017.png']
# B: ['img/17H-0863-1_L0003/L0003.png', 'img/17H-0863-1_L0005/L0005.png', 'img/17H-0863-1_L0009/L0009.png']
# C: ['img/17H-0863-1_L0002-old/L0002.png', 'img/17H-0863-1_L0008/L0008.png', 'img/17H-0863-1_L0013/L0013.png']
# D: ['img/17H-0863-1_L0007/L0007.png', 'img/17H-0863-1_L0012/L0012.png', 'img/17H-0863-1_L0014/L0014.png']
# E: ['img/17H-0863-1_L0004/L0004.png', 'img/17H-0863-1_L0015/L0015.png', 'img/17H-0863-1_L0016/L0016.png']
# F: ['img/17H-0863-1_L0001/L0001.png', 'img/17H-0863-1_L0010/L0010.png', 'img/17H-0863-1_L0018/L0018.png']

IMAGE_PATHS = [
    ['img/17H-0863-1_L0006-new/L0006.png', 'img/17H-0863-1_L0011/L0011.png', 'img/17H-0863-1_L0017/L0017.png'],
    ['img/17H-0863-1_L0003/L0003.png', 'img/17H-0863-1_L0005/L0005.png', 'img/17H-0863-1_L0009/L0009.png'],
    ['img/17H-0863-1_L0002-old/L0002.png', 'img/17H-0863-1_L0008/L0008.png', 'img/17H-0863-1_L0013/L0013.png'],
    ['img/17H-0863-1_L0007/L0007.png', 'img/17H-0863-1_L0012/L0012.png', 'img/17H-0863-1_L0014/L0014.png'],
    ['img/17H-0863-1_L0004/L0004.png', 'img/17H-0863-1_L0015/L0015.png', 'img/17H-0863-1_L0016/L0016.png'],
    ['img/17H-0863-1_L0001/L0001.png', 'img/17H-0863-1_L0010/L0010.png', 'img/17H-0863-1_L0018/L0018.png'],
]

LABEL_SET = [
    # [2,3,4,5,],
    # [0,3,4,5,],
    # [0,1,4,5,],
    # [0,1,2,5,],
    # [0,1,2,3,],
    # [1,2,3,4,],
    [0,1,2,3,4,5],
]

# LABELS = ['A', 'B', 'C', 'D', 'E', 'F']
LABELS = ['ALL']
label = 0
label_str = 'A'
args = sys.argv
DATASET_PATH = args[1]

for s in range(len(LABEL_SET)):
    label = s
    label_str = LABELS[s]
    R_data = np.array([])
    G_data = np.array([])
    B_data = np.array([])
    with tqdm(total=12, desc=label_str) as pbar:
        for i in range(len(LABEL_SET)):
            if i not in LABEL_SET[s]:
                continue
            for p in IMAGE_PATHS[i]:
                path = DATASET_PATH + '/' + p
                img_byte = tf.io.read_file(path)
                in_img = tf.image.decode_png(img_byte, channels=3).numpy()
                R_data = np.concatenate([R_data, in_img[:,:,0].ravel() ])
                G_data = np.concatenate([G_data, in_img[:,:,1].ravel() ])
                B_data = np.concatenate([B_data, in_img[:,:,2].ravel() ])
                pbar.update(1)  # プロセスバーを進行
    R_std = np.std(R_data)
    R_mean = np.mean(R_data)
    print(R_std)
    print(R_mean)

    G_std = np.std(G_data)
    G_mean = np.mean(G_data)
    print(G_std)
    print(G_mean)

    B_std = np.std(B_data)
    B_mean = np.mean(B_data)
    print(B_std)
    print(B_mean)

    with open('std-'+label_str+'.csv', 'w') as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerow(['R_std', 'R_mean', 'G_std', 'G_mean', 'B_std', 'B_mean'])
        writer.writerow([R_std, R_mean, G_std, G_mean, B_std, B_mean])
