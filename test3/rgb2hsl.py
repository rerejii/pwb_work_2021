import cv2
import numpy as np

path = 'img.png'
img = cv2.imread(path)
height = img.shape[0]
width = img.shape[1]
out = np.zeros([height, width, 3])

img_hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
out[:, :, 1:3] += img_hls[:, :, 1:3]
cv2.imwrite('img_hls.png', out)