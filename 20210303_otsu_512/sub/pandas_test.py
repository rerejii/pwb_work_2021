import pandas as pd
import numpy as np
import csv

# df = pd.read_csv('E:/work/myTensor/dataset2/DR-256-valid0-more/A/Property-A.csv')
# print(df)

with open('test.csv', 'w') as f:
    writer = csv.writer(f, lineterminator='\n')
    writer.writerow(['a', 'b', 'c'])
    writer.writerow([1,2,3])
    writer.writerow([4,5,6])
    writer.writerow([1,2,3])
    writer.writerow([1,2,3])

# print(df['a'])

df = pd.read_csv('test.csv')
print(df)
print(df['a'].values)
li = df['a'].values
print( 2 in li )
# print(df['a'].values[-1])
# a = np.average(np.array(df['a'].values))
# print(a)
# print(len(df.index))
# print(len(df.columns))
# print(df['a'].idxmax())

# with open('test.csv', 'r') as f:
