import cv2
import numpy as np

img_path = 'Z:/hayakawa/work20/tensor/test/crop_img/L12/otsu_l12_2.png'
ans_path = 'Z:/hayakawa/work20/tensor/test/crop_img/L12/ans_l12.png'
# otsu_l12_2

ori_img = cv2.imread(img_path)
ori_mask = cv2.imread(ans_path, 0)
rows,cols = ori_mask.shape

# マスクを左にずらす10
for i in range(20):
    img = np.array(ori_img)
    mask = np.array(ori_mask)
    M = np.float32([[1,0,i*-1],[0,1,-4]]) # x y
    mask = cv2.warpAffine(mask,M,(cols,rows))
    inv_mask = cv2.bitwise_not(mask)


    dst = cv2.distanceTransform(mask, cv2.DIST_L1, 5)
    img[dst==1] = [0, 255, 0]

    inv_dst = cv2.distanceTransform(inv_mask, cv2.DIST_L1, 5)
    img[inv_dst==1] = [255, 0, 0]

    cv2.imwrite('out_x2/out_x'+str(i)+'.png',img)