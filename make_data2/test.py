import numpy as np
import cv2
import sys

SAMPLE_FILE = 'E:/work/myTensor/dataset/img/17H-0863-1_L0001/L0001.png'
ANSWER_FILE = 'E:/work/myTensor/dataset/img/17H-0863-1_L0001/L0001_bin.png'

# SAMPLE_FILE = 'E:/work/myTensor/dataset/img/17H-0863-1_L0001/L0001/L0001_X007Y010.png'
# ANSWER_FILE = 'E:/work/myTensor/dataset/img/17H-0863-1_L0001/L0001_bin/L0001_bin_X007Y010.png'
OUT_FILE = 'L1.png'

sample_img = cv2.imread(SAMPLE_FILE)
sample_img = np.array(sample_img, dtype=np.uint8)
sample_img = cv2.cvtColor(sample_img, cv2.COLOR_BGR2RGB)
height, width, channel = sample_img.shape
print(channel)

mask = np.array((cv2.imread(ANSWER_FILE, 0) / 255) > 0.5)
result = np.zeros(sample_img.shape)

def ave_color(mask, img, mask_repeat=1):
    mask3 = mask.repeat(3).reshape(sample_img.shape)
    pass_img = np.array(img * mask3, dtype=np.uint8)
    count_n = np.sum(mask)
    # tmp = pass_img[:, :, 0].flatten()
    # tmp2 = [n for n in tmp if n != 0]
    # print(tmp2)
    # sys.exit()
    # li = [n for n in pass_img[:, :, 0]]
    # print(type(np.sum(pass_img[:, :, 0])))
    print(np.sum(pass_img[:, :, 0], dtype=np.uint64))
    print(np.sum(pass_img[:, :, 0], dtype=np.uint32))
    R = round(np.sum(pass_img[:, :, 0]) / count_n)
    G = round(np.sum(pass_img[:, :, 1]) / count_n)
    B = round(np.sum(pass_img[:, :, 2]) / count_n)
    print(np.sum(pass_img[:, :, 0]))
    print(count_n)
    print('%d %d %d' % (R, G, B))
    color = [R, G, B]
    return np.array(mask3 * color)

result += ave_color(mask==True, sample_img)
result = cv2.cvtColor(np.array(result, dtype=np.uint8), cv2.COLOR_RGB2BGR)
cv2.imwrite(OUT_FILE, result)

