import cv2
import math

img_path = 'Z:/hayakawa/work20/dataset2/img/17H-0863-1_L0012/L0012.png'
# ans_path = 'Z:/hayakawa/work20/dataset2/img/17H-0863-1_L0012/L0012_bin.png'
ans_path = 'C:/Users/hayakawa/work/mytensor/dataset2/img/17H-0863-1_L0012/52_L0012_bin.png'
out_ans_path = 'C:/Users/hayakawa/work/mytensor/dataset2/img/17H-0863-1_L0012/L0012_bin.png'

img = cv2.imread(img_path)
mask = cv2.imread(ans_path, 0)
mask = mask[52:-52, :]
cv2.imwrite(out_ans_path,mask)

# print(img.shape)
# print(mask.shape)
#
# mask = mask[52:-52, :]
# inv_mask = cv2.bitwise_not(mask)
#
# dst = cv2.distanceTransform(mask, cv2.DIST_L1, 5)
# img[dst==1] = [0, 255, 0]
# inv_dst = cv2.distanceTransform(inv_mask, cv2.DIST_L1, 5)
# img[inv_dst == 1] = [255, 0, 0]
#
# height, width, _ = img.shape
# half_h = math.floor(height / 2)
# half_w = math.floor(width / 2)
# img2 = img[0: half_h, 0: half_w]
#
# height, width, _ = img2.shape
# half_h = math.floor(height / 2)
# half_w = math.floor(width / 2)
# cut_out_A = img2[0: half_h, 0: half_w]
# cut_out_B = img2[0: half_h, half_w:]
# cut_out_C = img2[half_h:, 0: half_w]
# cut_out_D = img2[half_h:, half_w:]
#
# cv2.imwrite('out_bic_A.png',cut_out_A)
# cv2.imwrite('out_bic_B.png',cut_out_B)
# cv2.imwrite('out_bic_C.png',cut_out_C)
# cv2.imwrite('out_bic_D.png',cut_out_D)