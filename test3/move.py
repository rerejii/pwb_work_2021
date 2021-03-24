import cv2
import numpy as np

img_path = 'Z:/hayakawa/work20/tensor/test/crop_img/L12/sam/sum_l12.png'
ans_path = 'Z:/hayakawa/work20/tensor/test/crop_img/L12/ans_l12.png'

img = cv2.imread(img_path)
mask = cv2.imread(ans_path, 0)
rows,cols = mask.shape

# マスクを左にずらす10
M = np.float32([[1,0,0],[0,1,0]]) # x y
mask = cv2.warpAffine(mask,M,(cols,rows))
inv_mask = cv2.bitwise_not(mask)


dst = cv2.distanceTransform(mask, cv2.DIST_L1, 5)
img[dst==1] = [0, 255, 0]

inv_dst = cv2.distanceTransform(inv_mask, cv2.DIST_L1, 5)
img[inv_dst==1] = [255, 0, 0]

cv2.imwrite('out.png', img)
# cv2.imwrite('mask.png', mask)