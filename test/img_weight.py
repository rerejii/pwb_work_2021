import cv2
import numpy as np
import glob
import os
from natsort import natsorted
import sys

OUT_PATH = 'E:/work/myTensor/dataset2/'
# OUT_PATH = '/home/hayakawa/work/myTensor/dataset2/'
# OUT_PATH = 'C:/Users/hayakawa/work/mytensor/dataset2/'
origin_folder = OUT_PATH + 'img/*/*.png'
deep_folder = OUT_PATH + 'img/*/*_deep_19.png'


in_img_path = [path for path in glob.glob(origin_folder) if len(os.path.basename(path)) is 9]
in_img_path = natsorted(in_img_path)
deep_img_path = glob.glob(deep_folder)
deep_img_path = natsorted(deep_img_path)

for i in range(len(in_img_path)):
    print(in_img_path[i])
    sam_img = cv2.imread(in_img_path[i])
    sam_img = cv2.cvtColor(sam_img, cv2.COLOR_BGR2RGB)
    deep_img = cv2.imread(deep_img_path[i])
    deep_img = cv2.cvtColor(deep_img, cv2.COLOR_BGR2RGB)

    out_path = os.path.dirname(in_img_path[i]) + '/sub100.png'
    sub = np.abs(sam_img - deep_img)
    sub = np.array(sub, np.float32)
    sub = sub - 100
    sub[sub < 0] = 0
    print(sub)
    sub = cv2.cvtColor(sub, cv2.COLOR_RGB2BGR)
    cv2.imwrite(out_path, sub)

    print('mean==================')
    print(np.mean(sub))
    print('sum==================')
    print(np.sum(sub))
    print('max==================')
    print(np.max(sub))