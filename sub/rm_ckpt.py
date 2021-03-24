import os
import numpy as np
import sys
import glob
import shutil
from tqdm import tqdm

# args = sys.argv
# root_path = args[1]
root = 'Z:/hayakawa/binary/'
# root_path_set = ['20200927', '20201027', '20201124', '20201126', '20201208', '20201210', '20201211',
#                  '20201214', '20201216', '20201218', '20210115', '20210122', '20210126']
root_path_set = ['20210128']
root_path_set = [root + root_path_set[i] for i in range(len(root_path_set))]
for root_path in root_path_set:
    print(root_path)
    ckpt_path_set = glob.glob(root_path + '/**/**/CheckpointModel')
    pbar = tqdm(total=len(ckpt_path_set))
    for ckpt_path in ckpt_path_set:
        shutil.rmtree(ckpt_path)
        pbar.update(1)