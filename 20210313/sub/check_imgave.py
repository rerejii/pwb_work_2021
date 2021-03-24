import cv2
import csv
import tensorflow as tf
import numpy as np
from tqdm import tqdm
import os

# path
IMAGE_DIR = 'E:/work/myTensor/dataset2'

# 画像
IMAGE_SET = ['img/17H-0863-1_L0001/L0001.png',
             'img/17H-0863-1_L0002-old/L0002.png',
             'img/17H-0863-1_L0003/L0003.png',
             'img/17H-0863-1_L0004/L0004.png',
             'img/17H-0863-1_L0005/L0005.png',
             'img/17H-0863-1_L0006-new/L0006.png',
             'img/17H-0863-1_L0007/L0007.png',
             'img/17H-0863-1_L0008/L0008.png',
             'img/17H-0863-1_L0009/L0009.png',
             'img/17H-0863-1_L0010/L0010.png',
             'img/17H-0863-1_L0011/L0011.png',
             'img/17H-0863-1_L0012/L0012.png',
             'img/17H-0863-1_L0013/L0013.png',
             'img/17H-0863-1_L0014/L0014.png',
             'img/17H-0863-1_L0015/L0015.png',
             'img/17H-0863-1_L0016/L0016.png',
             'img/17H-0863-1_L0017/L0017.png',
             'img/17H-0863-1_L0018/L0018.png',]

# pathの合成
IMAGE_SET = [IMAGE_DIR +'/'+ s for s in IMAGE_SET]

with open('img-std.csv', 'w') as f:
    writer = csv.writer(f, lineterminator='\n')
    writer.writerow(['img_name', 'R_std', 'R_mean', 'G_std', 'G_mean', 'B_std', 'B_mean'])
    for i in tqdm(range(len(IMAGE_SET)), desc='Processed'):
        path = IMAGE_SET[i]
        img_byte = tf.io.read_file(path)
        in_img = tf.image.decode_png(img_byte, channels=3).numpy()
        R_data = in_img[:,:,0].ravel()
        G_data = in_img[:,:,1].ravel()
        B_data = in_img[:,:,2].ravel()

        R_std = np.std(R_data)
        R_mean = np.mean(R_data)
        G_std = np.std(G_data)
        G_mean = np.mean(G_data)
        B_std = np.std(B_data)
        B_mean = np.mean(B_data)

        img_name = os.path.splitext(os.path.basename(path))[0]

        writer.writerow([img_name ,R_std, R_mean, G_std, G_mean, B_std, B_mean])
