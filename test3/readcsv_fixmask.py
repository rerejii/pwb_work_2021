import cv2
import pandas as pd
import csv
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

filename = 'maskfix.csv'
df = pd.read_csv(filename)
data_list = np.zeros([41, 41])

for i in range(len(df.index)):
    move_h = df.at[i, 'move_h'] + 20
    move_w = df.at[i, 'move_w'] + 20
    data_list[move_h, move_w] += 1
# print(data_list)
# np.savetxt('count_maskfix.csv', data_list)
# data_list[21, 21]
# for i in range(41):
#     print('============')
#     print(i)
#     print(data_list[i, i])
data_list[20, 20] = 0
sns.heatmap(data_list, annot=False)
x = list(range(0, 41, 4))
val = list(range(-20, 21, 4))
plt.xticks(x, val)
plt.yticks(x, val)
plt.xlabel("shift width")
plt.ylabel("shift height")
plt.savefig('heatmap.png')