import numpy as np
import cv2

# sample_file = 'E:/work/myTensor/dataset2/img/17H-0863-1_L0003/L0003.png'
# answer_file = 'E:/work/myTensor/dataset2/img/17H-0863-1_L0003/L0003_bin.png'
# out_file = 'E:/work/myTensor/dataset2/img/17H-0863-1_L0003/L0003_deep.png'

ROOT_PATH = 'E:/work/myTensor/dataset2/img/'
SAMPLE_FILE_SET = [
                   '17H-0863-1_L0001/L0001.png', '17H-0863-1_L0002-old/L0002.png', '17H-0863-1_L0003/L0003.png',
                   '17H-0863-1_L0004/L0004.png', '17H-0863-1_L0005/L0005.png',
                   '17H-0863-1_L0006-new/L0006.png', '17H-0863-1_L0007/L0007.png', '17H-0863-1_L0008/L0008.png',
                   '17H-0863-1_L0009/L0009.png', '17H-0863-1_L0010/L0010.png', '17H-0863-1_L0011/L0011.png',
                   '17H-0863-1_L0012/L0012.png', '17H-0863-1_L0013/L0013.png', '17H-0863-1_L0014/L0014.png',
                   '17H-0863-1_L0015/L0015.png', '17H-0863-1_L0016/L0016.png', '17H-0863-1_L0017/L0017.png',
                   '17H-0863-1_L0018/L0018.png',
                   ]
ANSWER_FILE_SET = [
                   '17H-0863-1_L0001/L0001_bin.png', '17H-0863-1_L0002-old/L0002_bin.png', '17H-0863-1_L0003/L0003_bin.png',
                   '17H-0863-1_L0004/L0004_bin.png', '17H-0863-1_L0005/L0005_bin.png',
                   '17H-0863-1_L0006-new/L0006_bin.png', '17H-0863-1_L0007/L0007_bin.png', '17H-0863-1_L0008/L0008_bin.png',
                   '17H-0863-1_L0009/L0009_bin.png', '17H-0863-1_L0010/L0010_bin.png', '17H-0863-1_L0011/L0011_bin.png',
                   '17H-0863-1_L0012/L0012_bin.png', '17H-0863-1_L0013/L0013_bin.png', '17H-0863-1_L0014/L0014_bin.png',
                   '17H-0863-1_L0015/L0015_bin.png', '17H-0863-1_L0016/L0016_bin.png', '17H-0863-1_L0017/L0017_bin.png',
                   '17H-0863-1_L0018/L0018_bin.png',
                   ]
OUTPUT_FILE_SET = [
                   '17H-0863-1_L0001/L0001_deep.png', '17H-0863-1_L0002-old/L0002_deep.png', '17H-0863-1_L0003/L0003_deep.png',
                   '17H-0863-1_L0004/L0004_deep.png', '17H-0863-1_L0005/L0005_deep.png',
                   '17H-0863-1_L0006-new/L0006_deep.png', '17H-0863-1_L0007/L0007_deep.png', '17H-0863-1_L0008/L0008_deep.png',
                   '17H-0863-1_L0009/L0009_deep.png', '17H-0863-1_L0010/L0010_deep.png', '17H-0863-1_L0011/L0011_deep.png',
                   '17H-0863-1_L0012/L0012_deep.png', '17H-0863-1_L0013/L0013_deep.png', '17H-0863-1_L0014/L0014_deep.png',
                   '17H-0863-1_L0015/L0015_deep.png', '17H-0863-1_L0016/L0016_deep.png', '17H-0863-1_L0017/L0017_deep.png',
                   '17H-0863-1_L0018/L0018_deep.png',
                   ]
SAMPLE_FILE_SET = [ROOT_PATH + path for path in SAMPLE_FILE_SET]
ANSWER_FILE_SET = [ROOT_PATH + path for path in ANSWER_FILE_SET]
OUTPUT_FILE_SET = [ROOT_PATH + path for path in OUTPUT_FILE_SET]

def check_brank(img):
    height, width = img.shape
    top, bottom, left, right = 0, 0, 0, 0
    top_brank, bottom_brank, left_brank, right_brank = 0, 0, 0, 0
    # top
    for i in range(height):
        read = i
        line = img[read:read + 1, :]
        # val = np.sum(line)
        val = np.mean(line)
        if val > 120:
            top = read
            top_brank = read
            break

    # bottom
    for i in range(height):
        read = height - i - 1
        line = img[read:read + 1, :]
        # val = np.sum(line)
        val = np.mean(line)
        if val > 120:
            bottom = read
            bottom_brank = height - read - 1
            break

    # left
    for i in range(width):
        read = i
        line = img[:, read:read + 1]
        # val = np.sum(line)
        val = np.mean(line)
        if val > 120:
            left = read
            left_brank = read
            break

    # right
    for i in range(width):
        read = width - i - 1
        line = img[:, read:read + 1]
        # val = np.sum(line)
        val = np.mean(line)
        if val > 120:
            right = read
            right_brank = width - read - 1
            break

    brank = [top_brank, bottom_brank, left_brank, right_brank]
    return brank

for i in range(len(SAMPLE_FILE_SET)):
    print(SAMPLE_FILE_SET[i])

    sample_file = SAMPLE_FILE_SET[i]
    answer_file = ANSWER_FILE_SET[i]
    out_file = OUTPUT_FILE_SET[i]

    sample_img = cv2.imread(sample_file)
    sample_img = np.array(sample_img)
    height, width, channel = sample_img.shape

    answer_img = cv2.imread(answer_file, 0)
    mask = answer_img / 255
    mask = np.array(mask)
    top_brank, bottom_brank, left_brank, right_brank = check_brank(answer_img)

    huti_out_mask = np.zeros(shape=(height, width))
    huti_out_mask[top_brank: -bottom_brank, left_brank: -right_brank] = np.ones(
        shape=(height - top_brank - bottom_brank, width - left_brank - right_brank))

    sample_gray = cv2.cvtColor(sample_img, cv2.COLOR_BGR2GRAY)
    top_brank, bottom_brank, left_brank, right_brank = check_brank(sample_gray)
    huti_b_mask = np.zeros(shape=(height, width))
    huti_b_mask[top_brank: -bottom_brank, left_brank: -right_brank] = np.ones(
        shape=(height - top_brank - bottom_brank, width - left_brank - right_brank))




    onemask = mask * huti_out_mask
    zeromask = (mask == 0) * huti_out_mask
    huti_color_mask = (mask == 0) * huti_b_mask * (huti_out_mask == 0)
    huti_brack_mask = (mask == 0) * (huti_b_mask == 0) * (huti_out_mask == 0)

    onemask3 = onemask.repeat(3)
    onemask3 = onemask3.reshape([height, width, channel])
    count_onemask = np.sum(onemask)
    cut_oneimg = sample_img * onemask3
    one_B = round(np.sum(cut_oneimg[:, :, 0] / count_onemask))
    one_G = round(np.sum(cut_oneimg[:, :, 1] / count_onemask))
    one_R = round(np.sum(cut_oneimg[:, :, 2] / count_onemask))
    one_color = [one_B, one_G, one_R]
    result = (onemask3 * one_color)

    del onemask3
    del cut_oneimg

    zeromask3 = zeromask.repeat(3)
    zeromask3 = zeromask3.reshape([height, width, channel])
    cut_zeroimg = sample_img * zeromask3
    count_zeromask = np.sum(zeromask)
    zero_B = round(np.sum(cut_zeroimg[:, :, 0] / count_zeromask))
    zero_G = round(np.sum(cut_zeroimg[:, :, 1] / count_zeromask))
    zero_R = round(np.sum(cut_zeroimg[:, :, 2] / count_zeromask))
    zero_color = [zero_B, zero_G, zero_R]
    result += (zeromask3 * zero_color)

    del zeromask3
    del cut_zeroimg

    if np.sum(huti_color_mask) is not 0:
        huti_color_mask3 = huti_color_mask.repeat(3)
        huti_color_mask3 = huti_color_mask3.reshape([height, width, channel])
        count_huti_color_mask = np.sum(huti_color_mask)
        cut_huti_color_img = sample_img * huti_color_mask3
        one_B = round(np.sum(cut_huti_color_img[:, :, 0] / count_huti_color_mask))
        one_G = round(np.sum(cut_huti_color_img[:, :, 1] / count_huti_color_mask))
        one_R = round(np.sum(cut_huti_color_img[:, :, 2] / count_huti_color_mask))
        huti_color_color = [one_B, one_G, one_R]
        result += (huti_color_mask3 * huti_color_color)

        del huti_color_mask3
        del cut_huti_color_img

    huti_brack_mask3 = huti_brack_mask.repeat(3)
    huti_brack_mask3 = huti_brack_mask3.reshape([height, width, channel])
    count_huti_brack_mask = np.sum(huti_brack_mask)
    cut_huti_brack_img = sample_img * huti_brack_mask3
    one_B = round(np.sum(cut_huti_brack_img[:, :, 0] / count_huti_brack_mask))
    one_G = round(np.sum(cut_huti_brack_img[:, :, 1] / count_huti_brack_mask))
    one_R = round(np.sum(cut_huti_brack_img[:, :, 2] / count_huti_brack_mask))
    huti_brack_color = [one_B, one_G, one_R]
    result += (huti_brack_mask3 * huti_brack_color)

    del huti_brack_mask3
    del cut_huti_brack_img

    cv2.imwrite(out_file, result)

