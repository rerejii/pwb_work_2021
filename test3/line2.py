import cv2

# ROOT_PATH = 'C:/Users/hayakawa/work/mytensor/dataset2/Img-256-valid128-more/'
ROOT_PATH = '/localhome/rerejii/work/myTensor/dataset2/Img-256-valid128-more/'
CHECK_SAMPLE_PATH = [
    '17H-0863-1_L0001/Sample/Sample_X009_Y038.png',
    '17H-0863-1_L0002-old/Sample/Sample_X002_Y032.png',
    '17H-0863-1_L0003/Sample/Sample_X004_Y035.png',
    '17H-0863-1_L0004/Sample/Sample_X008_Y039.png',
    '17H-0863-1_L0005/Sample/Sample_X009_Y008.png',
    '17H-0863-1_L0006-new/Sample/Sample_X002_Y015.png',
    '17H-0863-1_L0007/Sample/Sample_X018_Y006.png',
    '17H-0863-1_L0008/Sample/Sample_X007_Y026.png',
    '17H-0863-1_L0009/Sample/Sample_X018_Y067.png',
    '17H-0863-1_L0010/Sample/Sample_X015_Y064.png',
    '17H-0863-1_L0011/Sample/Sample_X017_Y052.png',
    '17H-0863-1_L0012/Sample/Sample_X005_Y033.png',
    '17H-0863-1_L0013/Sample/Sample_X008_Y074.png',
    '17H-0863-1_L0014/Sample/Sample_X002_Y069.png',
    '17H-0863-1_L0015/Sample/Sample_X005_Y004.png',
    '17H-0863-1_L0016/Sample/Sample_X003_Y036.png',
    '17H-0863-1_L0017/Sample/Sample_X009_Y071.png',
    '17H-0863-1_L0018/Sample/Sample_X037_Y022.png',
]
CHECK_ANSWER_PATH = [
    '17H-0863-1_L0001/Answer/Answer_X009_Y038.png',
    '17H-0863-1_L0002-old/Answer/Answer_X002_Y032.png',
    '17H-0863-1_L0003/Answer/Answer_X004_Y035.png',
    '17H-0863-1_L0004/Answer/Answer_X008_Y039.png',
    '17H-0863-1_L0005/Answer/Answer_X009_Y008.png',
    '17H-0863-1_L0006-new/Answer/Answer_X002_Y015.png',
    '17H-0863-1_L0007/Answer/Answer_X018_Y006.png',
    '17H-0863-1_L0008/Answer/Answer_X007_Y026.png',
    '17H-0863-1_L0009/Answer/Answer_X018_Y067.png',
    '17H-0863-1_L0010/Answer/Answer_X015_Y064.png',
    '17H-0863-1_L0011/Answer/Answer_X017_Y052.png',
    '17H-0863-1_L0012/Answer/Answer_X005_Y033.png',
    '17H-0863-1_L0013/Answer/Answer_X008_Y074.png',
    '17H-0863-1_L0014/Answer/Answer_X002_Y069.png',
    '17H-0863-1_L0015/Answer/Answer_X005_Y004.png',
    '17H-0863-1_L0016/Answer/Answer_X003_Y036.png',
    '17H-0863-1_L0017/Answer/Answer_X009_Y071.png',
    '17H-0863-1_L0018/Answer/Answer_X037_Y022.png',
]

CHECK_SAMPLE_PATH = [ROOT_PATH + CHECK_SAMPLE_PATH[i] for i in range(18)]
CHECK_ANSWER_PATH = [ROOT_PATH + CHECK_ANSWER_PATH[i] for i in range(18)]

for i in range(18):
    img_path = CHECK_SAMPLE_PATH[i]
    ans_path = CHECK_ANSWER_PATH[i]
    img = cv2.imread(img_path)
    mask = cv2.imread(ans_path, 0)
    inv_mask = cv2.bitwise_not(mask)

    dst = cv2.distanceTransform(mask, cv2.DIST_L1, 5)
    img[dst==1] = [0, 255, 0]

    inv_dst = cv2.distanceTransform(inv_mask, cv2.DIST_L1, 5)
    img[inv_dst==1] = [255, 0, 0]

    cv2.imwrite('leo3_out/' + str(i+1) + '.png',img)