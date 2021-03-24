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
                   '17H-0863-1_L0001/L0001_deep_19.png', '17H-0863-1_L0002-old/L0002_deep_19.png', '17H-0863-1_L0003/L0003_deep_19.png',
                   '17H-0863-1_L0004/L0004_deep_19.png', '17H-0863-1_L0005/L0005_deep_19.png',
                   '17H-0863-1_L0006-new/L0006_deep_19.png', '17H-0863-1_L0007/L0007_deep_19.png', '17H-0863-1_L0008/L0008_deep_19.png',
                   '17H-0863-1_L0009/L0009_deep_19.png', '17H-0863-1_L0010/L0010_deep_19.png', '17H-0863-1_L0011/L0011_deep_19.png',
                   '17H-0863-1_L0012/L0012_deep_19.png', '17H-0863-1_L0013/L0013_deep_19.png', '17H-0863-1_L0014/L0014_deep_19.png',
                   '17H-0863-1_L0015/L0015_deep_19.png', '17H-0863-1_L0016/L0016_deep_19.png', '17H-0863-1_L0017/L0017_deep_19.png',
                   '17H-0863-1_L0018/L0018_deep_19.png',
                   ]
SAMPLE_FILE_SET = [ROOT_PATH + path for path in SAMPLE_FILE_SET]
ANSWER_FILE_SET = [ROOT_PATH + path for path in ANSWER_FILE_SET]
OUTPUT_FILE_SET = [ROOT_PATH + path for path in OUTPUT_FILE_SET]

def check_brank(img):
    print('check brank process now')
    height, width = img.shape
    top, bottom, left, right = 0, 0, 0, 0
    top_brank, bottom_brank, left_brank, right_brank = 0, 0, 0, 0
    # top
    for i in range(height):
        read = i
        line = img[read:read + 1, :]
        val = np.sum(line)
        if val != 0:
            top = read
            top_brank = read
            break

    # bottom
    for i in range(height):
        read = height - i - 1
        line = img[read:read + 1, :]
        val = np.sum(line)
        if val != 0:
            bottom = read
            bottom_brank = height - read - 1
            break

    # left
    for i in range(width):
        read = i
        line = img[:, read:read + 1]
        val = np.sum(line)
        if val != 0:
            left = read
            left_brank = read
            break

    # right
    for i in range(width):
        read = width - i - 1
        line = img[:, read:read + 1]
        val = np.sum(line)
        if val != 0:
            right = read
            right_brank = width - read - 1
            break

    brank = [top_brank, bottom_brank, left_brank, right_brank]
    print('check brank process end')
    return brank

for i in range(len(SAMPLE_FILE_SET)):
    print(SAMPLE_FILE_SET[i])

# for i in [1, 2]:
#     print(SAMPLE_FILE_SET[i])

    sample_file = SAMPLE_FILE_SET[i]
    answer_file = ANSWER_FILE_SET[i]
    out_file = OUTPUT_FILE_SET[i]

    sample_img = cv2.imread(sample_file)
    sample_img = np.array(sample_img)
    height, width, channel = sample_img.shape

    mask = cv2.imread(answer_file, 0) / 255
    mask = np.array(mask)
    top_brank, bottom_brank, left_brank, right_brank = check_brank(mask)

    hutimask = np.zeros(shape=(height, width))
    hutimask[top_brank: -bottom_brank, left_brank : -right_brank] = np.ones(shape=(height-top_brank-bottom_brank, width-left_brank-right_brank))

    onemask = mask * hutimask
    onemask3 = onemask.repeat(3)
    onemask3 = onemask3.reshape([height, width, channel])
    cut_oneimg = sample_img * onemask3
    count_onemask = np.sum(onemask)
    one_B = round(np.sum(cut_oneimg[:, :, 0] / count_onemask))
    one_G = round(np.sum(cut_oneimg[:, :, 1] / count_onemask))
    one_R = round(np.sum(cut_oneimg[:, :, 2] / count_onemask))
    one_color = [one_B, one_G, one_R]
    result = (onemask3 * one_color)

    del onemask3
    del cut_oneimg
    del one_B
    del one_G
    del one_R
    del one_color
    del count_onemask
    print('one end')

    zeromask = (mask==0) * 1 * hutimask
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
    del zero_B
    del zero_G
    del zero_R
    del zero_color
    del count_zeromask
    print('zero end')

    outmask = (hutimask == 0) * (sample_img.sum(axis=2)!=0)
    outmask3 = outmask.repeat(3)
    outmask3 = outmask3.reshape([height, width, channel])
    cut_outimg = sample_img * outmask3
    count_outmask = np.sum(outmask)
    out_B = round(np.sum(cut_outimg[:, :, 0] / count_outmask))
    out_G = round(np.sum(cut_outimg[:, :, 1] / count_outmask))
    out_R = round(np.sum(cut_outimg[:, :, 2] / count_outmask))
    out_color = [out_B, out_G, out_R]
    result += (outmask3 * out_color)

    del outmask3
    del cut_outimg
    del out_B
    del out_G
    del out_R
    del out_color
    del count_outmask
    print('out end')

    del sample_img
    del mask

    cv2.imwrite(out_file, result)
