import pandas as pd
import os
import csv

df = pd.read_csv('img-std.csv')
header = ['R_std', 'R_mean', 'G_std', 'G_mean', 'B_std', 'B_mean']
print(df)

mean_vals = ['mean']
max_vals = ['max']
min_vals = ['min']

for s in header:
    mean_vals.append(df[s].mean())
    max_vals.append(df[s].max())
    min_vals.append(df[s].min())

with open('img-std.csv', 'a') as f:
    writer = csv.writer(f, lineterminator='\n')
    writer.writerow(mean_vals)
    writer.writerow(max_vals)
    writer.writerow(min_vals)
