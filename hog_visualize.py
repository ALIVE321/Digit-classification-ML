import os
import cv2
import pickle
from tqdm import tqdm
import numpy as np
from clf_lda import *
from clf_lr import *
import numpy as np
import pickle
import torch
import os
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os, sys
from scipy.signal import savgol_filter

sys.path.append("./clf_lda")
sys.path.append("./clf_lr")

winSize = (16, 16)
blockSize = (8, 8)
blockStride = (4, 4)
cellSize = (4, 4)
n_bins = 9
# hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, n_bins)

winStride = (10, 10)
padding = (4, 4)

# HOG: 9 * 9 * 4 * 9 = 2916

ws = [
    (-4, -4, 12, 12), (-4, 6, 12, 22), (-4, 16, 12, 32),
    (6, -4, 12, 12), (6, 6, 12, 22), (6, 16, 12, 32),
    (16, -4, 12, 12), (16, 6, 12, 22), (16, 16, 12, 32)
]

def compute(x):
    direction = x % 9
    cell = (x // 9) % 4 
    block =  (x // 36) % 9
    window = x // (9*9*4)
    wx0, wy0, wx1, wy1 = ws[window]
    b_d, b_m = block // 3, block % 3
    bx0, by0 = wx0 + b_d * 4, wy0 + b_m * 4
    bx1, by1 = bx0 + 8, by0 + 8
    c_d, c_m = cell // 2, cell % 2
    cx0, cy0 = bx0 + c_d * 4, by0 + c_d * 4
    cx1, cy1 = cx0 + 4, cy0 + 4
    return cx0, cy0, cx1, cy1
        


if __name__ == "__main__":
    sample_0_img = cv2.resize((cv2.imread("./data/train/4/13-1.png")), (32, 32))
    # sample_8_img = cv2.resize((cv2.imread("./data/train/8/7-2.png")), (32, 32))
    # features = hog.compute(sample_img, winStride, padding).reshape(-1)
    # print(features)

    lrl1_clfs = torch.load("./clf_lr/lr_lasso_1e-1.pickle")
    beta_0 = np.array(lrl1_clfs[4].beta.cpu())
    print(beta_0)
    # beta_8 = np.array(lrl1_clfs[8].beta.cpu())

    eps = 0.25

    for i, j in enumerate(beta_0):
        if j > eps:
            cell = compute(i)
            sample_0_img[cell[0]: cell[2]][cell[1]: cell[3]] = 255
        if j < -eps:
            cell = compute(i)
            sample_0_img[cell[0]: cell[2]][cell[1]: cell[3]] = 0

    # for i, j in enumerate(beta_8):
    #     if j > eps:
    #         cell = compute(i)
    #         sample_8_img[cell[0]: cell[2]][cell[1]: cell[3]] = 255
    #     if j < -eps:
    #         cell = compute(i)
    #         sample_8_img[cell[0]: cell[2]][cell[1]: cell[3]] = 0

    cv2.imwrite("./sample.png", sample_0_img)


    # with open("./data/test_hog.pickle", "rb") as f:
    #     test_data = pickle.load(f)
    # X_test = test_data["hog"]
    # y_test = test_data["label"]
    # del test_data
    # # print(X_test.shape, y_test.shape)

    # X, y = [], []
    # print(X)
    # for i, yy in enumerate(y_test):
    #     if yy == 8:
    #         X.append(X_test[i])
    #         y.append(0)
    # X = torch.tensor(X, device='cpu')
    # y = torch.tensor(y, device='cpu')

    # preds = lrl1_clfs[0].predict(X)
    # print(preds)

    # print(f"{torch.sum(preds > 0.5).item() / len(y):.4f}")

    