import os
import glob
import csv
import sys
import numpy as np
import pandas as pd
from natsort import natsorted
import matplotlib.pyplot as plt
# import matplotlib
mark = 'A'

csv_path = 'Z:/hayakawa/binary/20210227/unet_use-bias_beta/unet_use-bias_beta-'+mark+'/CsvDatas/train_loss.csv'
# csv_path = 'Z:/hayakawa/binary/20210227/unet_use-bias_beta_loss/unet_use-bias_beta_loss-A/CsvDatas/train_loss.csv'
df = pd.read_csv(csv_path)
# print(df)
df.plot(y=['loss'])
# print(df['loss'].values)
# print(df.index.values)
# x = df.index.values
# y = df['loss'].values
# plt.plot(x, y)
plt.ylim([0.01, 0.06])
plt.savefig('train_loss-'+mark+'.png')
plt.show()