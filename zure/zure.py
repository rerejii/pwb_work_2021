import cv2
import numpy as np
# E:/work/myTensor/dataset2/img
# Z:/hayakawa/share/img/

black = [0, 0, 0]  # 予測:0黒 解答:0黒
white = [255, 255, 255]  # 予測:1白 解答:1白
red = [255, 0, 0]  # 予測:1白 解答:0黒
blue = [0, 204, 255]  # 予測:0黒 解答:1白
green = [0, 255, 0]

img_path = 'sam_l11.png'
view_path = 'view_l11.png'
# img_path = 'E:/work/myTensor/dataset2/img/17H-0863-1_L0011/L0011.png'
# img_path = 'E:/work/myTensor/dataset2/img/17H-0863-1_L0012/L0012.png'
# img_path = 'E:/work/myTensor/dataset2/img/17H-0863-1_L0017/L0017.png'

mask_path = 'ans_l11.png'
# mask_path = 'E:/work/myTensor/dataset2/img/17H-0863-1_L0011/L0011_bin.png'
# mask_path = 'E:/work/myTensor/dataset2/img/17H-0863-1_L0012/L0012_bin.png'
# mask_path = 'E:/work/myTensor/dataset2/img/17H-0863-1_L0017/L0017_bin.png'

img = cv2.imread(img_path)
view = cv2.imread(view_path)
mask = cv2.imread(mask_path, 0)
h, w = mask.shape
dst = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
print(dst.shape)

bound = np.array((dst != 0) * (dst < 2), np.uint8)
# weight.numpy().repeat(3).reshape(256, 256, 3)
mask3 = mask.repeat(3).reshape(h, w, 3)
bound3 = bound.repeat(3).reshape(h, w, 3)

# result_img = np.zeros([h, w, 3])
# result_img += white * (mask == 255) * (bound3 == 0)

# out = img * 1.0 + mask3 * 0.0
out = np.zeros([h, w, 3])
out += img * (bound3 == 0)

cv2.imwrite('out_l11.png', out)

out2 = np.zeros([h, w, 3])
out2 += view * (bound3 == 0)
out2 += green * (bound3 == 1)
cv2.imwrite('out2_l11.png', out2)