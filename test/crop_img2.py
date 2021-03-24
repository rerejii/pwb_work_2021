import cv2
crop_size = 1024
# Z:/hayakawa/binary/20201218/unet_fin-none_logit-true_thr-d0/unet_fin-none_logit-true_thr-d0-A
# E:/work/myTensor/dataset2/img/17H-0863-1_L0011
# img = cv2.imread('Z:/hayakawa/binary/20201218/unet_fin-none_logit-true_thr-d0/unet_fin-none_logit-true_thr-d0-A/View_L0011.png')
# img = cv2.imread('E:/work/myTensor/dataset2/img/17H-0863-1_L0012/L0012.png')
# img = cv2.imread('E:/work/myTensor/dataset2/img/17H-0863-1_L0012/L0012_bin.png', 0)
# img = cv2.imread('Z:/hayakawa/binary/20210128/unet_use-bias_beta/unet_use-bias_beta-D/generator_L0012.png')
# img = cv2.imread('Z:/hayakawa/binary/20210128/unet_use-bias_beta_otsu2/unet_use-bias_beta_otsu2-D/generator_L0012.png')
# img = cv2.imread('Z:/hayakawa/work20/dataset2/img/17H-0863-1_L0012/L0012_hls_l_b.png', 0)
# img = cv2.imread('E:/work/myTensor/dataset2/img/17H-0863-1_L0004/L0004.png')
# img = cv2.imread('E:/work/myTensor/dataset2/img/17H-0863-1_L0004/L0004_bin.png', 0)
# img = cv2.imread('Z:/hayakawa/work20/dataset2/img/17H-0863-1_L0012/L0012_brank_otsu_l.png', 0)
# img = cv2.imread('Z:/hayakawa/binary/20210128/unet_ls_5/unet_ls_5-D/generator_L0012.png', 0)
# img = cv2.imread('Z:/hayakawa/binary/20210128/unet_use-bias_beta_otsu2_deep_multi-loss/unet_use-bias_beta_otsu2_deep_multi-loss-E/generator_L0004.png', 0)
# img = cv2.imread('Z:/hayakawa/binary/20210128/unet_use-bias_beta_otsu2/unet_use-bias_beta_otsu2-E/generator_L0004.png', 0)
# img = cv2.imread('Z:/hayakawa/binary/20210128/unet_ls_5/unet_ls_5-E/generator_L0004.png', 0)
# img = cv2.imread('Z:/hayakawa/binary/20210128/unet_use-bias_beta/unet_use-bias_beta-E/generator_L0004.png', 0)

img = cv2.imread('Z:/hayakawa/work20/dataset2/img/17H-0863-1_L0004/L0004_brank_otsu_l_c3.png', 0)

# img = cv2.imread('Z:/hayakawa/work20/dataset2/img/17H-0863-1_L0012/L0012_brank_otsu_l_c5.png', 0)
#'Z:/hayakawa/work20/dataset2/img/17H-0863-1_L0012/L0012_hls_l_b.png'
height, width = img.shape[:2]
# half_h = int(19608/8) * 7 - 300
# half_w = int(24576/4) * 1

# half_h = 512
# half_w = 512

# half_h = int(19608/4) * 3 - 700
# half_w = int(24576/16) * 1 + 400

# half_h = int(19608/2)
# half_w = int(24576/16)

half_h = int(19608/2)
half_w = int(24576/4)-1800

half_c = int(crop_size / 2)
crop_img = img[half_h-half_c: half_h+half_c, half_w-half_c: half_w+half_c]
# cv2.imwrite('crop_img/unet_otsu2.png', crop_img)
# cv2.imwrite('crop_img/unet_loop2.png', crop_img)
# cv2.imwrite('crop_img/sam_4_bin.png', crop_img)
cv2.imwrite('crop_img/otsu_closing_4.png', crop_img)

# img2 = cv2.imread('E:/work/myTensor/dataset2/img/17H-0863-1_L0012/L0012_bin.png', 0)
# crop_img = img2[half_h-half_c: half_h+half_c, half_w-half_c: half_w+half_c]
# cv2.imwrite('crop_img/ans_l12.png', crop_img)

# img2 = cv2.imread('E:/work/myTensor/dataset2/img/17H-0863-1_L0002-old/L0002_bin.png', 0)
# crop_img = img2[half_h-half_c: half_h+half_c, half_w-half_c: half_w+half_c]
# cv2.imwrite('crop_img/ans_l2_2.png', crop_img)
# #
# img2 = cv2.imread('Z:/hayakawa/binary/20210128/unet_use-bias_beta/unet_use-bias_beta-C/generator_L0002.png', 0)
# crop_img = img2[half_h-half_c: half_h+half_c, half_w-half_c: half_w+half_c]
# cv2.imwrite('crop_img/gen_l2_2.png', crop_img)