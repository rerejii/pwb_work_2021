import cv2
import numpy as np
import os
# Z:/hayakawa/binary/20210222/unet_use-bias_beta/unet_use-bias_beta-F/accuracy_set
# Z:/hayakawa/binary/20210222/unet_use-bias_beta_otsu/unet_use-bias_beta_otsu-F/accuracy_set
path = 'Z:/hayakawa/binary/20210222/unet_use-bias_beta_otsu/unet_use-bias_beta_otsu-F/accuracy_set/View_L0001.png'
img = cv2.imread(path)
height = img.shape[0]
width = img.shape[1]
img2 = cv2.resize(img, (int(width * 0.25), int(height * 0.25)))
cv2.imwrite(os.path.splitext(path)[0] + 'd25.png', img2)
