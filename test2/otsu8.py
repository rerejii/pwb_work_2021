import cv2
import numpy as np
import matplotlib
import os
from matplotlib import pyplot as plt
from tqdm import tqdm
from natsort import natsorted
import pandas as pd
import glob
import csv
import time as timefunc
import math
matplotlib.use('Agg')

IN_PATH = 'Z:/hayakawa/work20/dataset2/img/17H-0863-1_L0012/L0012.png'
OUT_PATH = 'Z:/hayakawa/work20/dataset2/img/17H-0863-1_L0012/L0012_hls_l_b.png'

img = cv2.imread(IN_PATH)
img_hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)

ret2_l, img_otsu_l = cv2.threshold(img_hls[:, :, 1], 0, 255, cv2.THRESH_OTSU)
cv2.imwrite(OUT_PATH, img_otsu_l)