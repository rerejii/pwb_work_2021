import cv2
import numpy as np
import csv
import os
from tqdm import tqdm
import sys
import pandas as pd
brank = [546+100, 347+100, 330+100, 319+100]
ANS_PATHSET = [
    'Z:/hayakawa/work20/dataset2/img/17H-0863-1_L0001/L0001_bin.png',
    'Z:/hayakawa/work20/dataset2/img/17H-0863-1_L0002-old/L0002_bin.png',
    'Z:/hayakawa/work20/dataset2/img/17H-0863-1_L0003/L0003_bin.png',
    'Z:/hayakawa/work20/dataset2/img/17H-0863-1_L0004/L0004_bin.png',
    'Z:/hayakawa/work20/dataset2/img/17H-0863-1_L0005/L0005_bin.png',
    'Z:/hayakawa/work20/dataset2/img/17H-0863-1_L0006-new/L0006_bin.png',
    'Z:/hayakawa/work20/dataset2/img/17H-0863-1_L0007/L0007_bin.png',
    'Z:/hayakawa/work20/dataset2/img/17H-0863-1_L0008/L0008_bin.png',
    'Z:/hayakawa/work20/dataset2/img/17H-0863-1_L0009/L0009_bin.png',
    'Z:/hayakawa/work20/dataset2/img/17H-0863-1_L0010/L0010_bin.png',
    'Z:/hayakawa/work20/dataset2/img/17H-0863-1_L0011/L0011_bin.png',
    'Z:/hayakawa/work20/dataset2/img/17H-0863-1_L0012/L0012_bin.png',
    'Z:/hayakawa/work20/dataset2/img/17H-0863-1_L0013/L0013_bin.png',
    'Z:/hayakawa/work20/dataset2/img/17H-0863-1_L0014/L0014_bin.png',
    'Z:/hayakawa/work20/dataset2/img/17H-0863-1_L0015/L0015_bin.png',
    'Z:/hayakawa/work20/dataset2/img/17H-0863-1_L0016/L0016_bin.png',
    'Z:/hayakawa/work20/dataset2/img/17H-0863-1_L0017/L0017_bin.png',
    'Z:/hayakawa/work20/dataset2/img/17H-0863-1_L0018/L0018_bin.png',
]
index = [os.path.basename(path) for path in ANS_PATHSET]

with tqdm(total=len(ANS_PATHSET), desc='Processed') as pbar:
    # writer = csv.writer(f, delimiter=',')
    # writer.writerows(['imgname', 'white', 'black'])
    df = pd.DataFrame(None,
                      columns=['white', 'black'],
                      index=index)
    for i in range(len(ANS_PATHSET)):
        ans = cv2.imread(ANS_PATHSET[i], 0)
        ans = ans[brank[0]:-1 * brank[1], brank[2]:-1 * brank[3]]
        inv_ans = cv2.bitwise_not(ans)
        w = np.average(ans/255)
        b = np.average(inv_ans/255)
        name = os.path.basename(ANS_PATHSET[i])
        df.loc[name] = [w, b]
        pbar.update(1)
    df.to_csv('brank_sirokuro.csv')
