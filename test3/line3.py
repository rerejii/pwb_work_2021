import cv2
import math
from natsort import natsorted
import glob
import os

root_img_path = 'C:/Users/hayakawa/work/mytensor/dataset2/Img-256-valid128-more/17H-0863-1_L0012/Sample'
root_ans_path = 'C:/Users/hayakawa/work/mytensor/dataset2/Img-256-valid128-more/17H-0863-1_L0012/Answer'
root_out_path = 'C:/Users/hayakawa/work/mytensor/dataset2/Img-256-valid128-more/17H-0863-1_L0012/maskline'
os.makedirs(root_out_path, exist_ok=True)
img_path_set = glob.glob(root_img_path + '/*.png')
img_path_set = natsorted(img_path_set)
ans_path_set = glob.glob(root_ans_path + '/*.png')
ans_path_set = natsorted(ans_path_set)

for i in range(len(img_path_set)):
# for i in range(10):

    img_path = img_path_set[i]
    ans_path = ans_path_set[i]
    out_path = root_out_path + '/maskline' + os.path.basename(img_path)[6:]
    img = cv2.imread(img_path)
    mask = cv2.imread(ans_path, 0)
    inv_mask = cv2.bitwise_not(mask)

    dst = cv2.distanceTransform(mask, cv2.DIST_L1, 5)
    img[dst==1] = [0, 255, 0]

    inv_dst = cv2.distanceTransform(inv_mask, cv2.DIST_L1, 5)
    img[inv_dst==1] = [255, 0, 0]

    cv2.imwrite(out_path, img)







# img_path = 'C:/Users/hayakawa/work/mytensor/dataset2/Img-256-valid0-more/17H-0863-1_L0012/Sample/Sample_X005_Y033.png'
# ans_path = 'leo3_in/Answer_X005_Y033.png'
#
# img = cv2.imread(img_path)
# mask = cv2.imread(ans_path, 0)
#
# dst = cv2.distanceTransform(mask, cv2.DIST_L1, 5)
#
# img[dst==1] = [0, 255, 0]
# cv2.imwrite('leo3.png', img)
# height, width, _ = img.shape
# half_h = math.floor(height / 2)
# half_w = math.floor(width / 2)
# cut_out_A = img[0: half_h, 0: half_w]
# cut_out_B = img[0: half_h, half_w:]
# cut_out_C = img[half_h:, 0: half_w]
# cut_out_D = img[half_h:, half_w:]
#
# cv2.imwrite('out_bic_A.png',cut_out_A)
# cv2.imwrite('out_bic_B.png',cut_out_B)
# cv2.imwrite('out_bic_C.png',cut_out_C)
# cv2.imwrite('out_bic_D.png',cut_out_D)