import cv2
import numpy as np
# Z:/hayakawa/work20/dataset2/img/17H-0863-1_L0012
img_path = 'Z:/hayakawa/work20/dataset2/img/17H-0863-1_L0012/L0012.png'
ans_path = 'Z:/hayakawa/work20/dataset2/img/17H-0863-1_L0012/L0012_bin_old.png'
out_path = 'Z:/hayakawa/work20/dataset2/img/17H-0863-1_L0012/L0012_bin.png'
out_path2 = 'Z:/hayakawa/work20/dataset2/img/17H-0863-1_L0012/L0012_zure.png'

img = cv2.imread(img_path)
mask = cv2.imread(ans_path, 0)
rows,cols = mask.shape

# マスクを左にずらす10
M = np.float32([[1,0,-10],[0,1,-5]]) # x y
mask = cv2.warpAffine(mask,M,(cols,rows))
inv_mask = cv2.bitwise_not(mask)


dst = cv2.distanceTransform(mask, cv2.DIST_L1, 5)
img[dst==1] = [0, 255, 0]

inv_dst = cv2.distanceTransform(inv_mask, cv2.DIST_L1, 5)
img[inv_dst==1] = [255, 0, 0]

cv2.imwrite(out_path2, img)
cv2.imwrite(out_path, mask)