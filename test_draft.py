# import cv2
import numpy as np
import pickle
import torch
import os
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from torch.utils.data import DataLoader

def img_show(imag, name="img", key=0):
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    # h, w = np.shape(imag)[:2]
    # h *= (300 / w)
    # h = int(h)
    # cv2.resizeWindow(name, 300, h)
    cv2.imshow(name, imag)
    cv2.waitKey(key)
    cv2.destroyAllWindows()

# with open("./data/test_hog.pickle", "rb") as f:
#     data = pickle.load(f)
# #
# for img in data["hog"]:
#     img_show(img)
def plt_fig(A, B, num):
    plt.plot(A, B)


with open("data/test_data.pickle", "rb") as f:
    test_data = pickle.load(f)

X_test = torch.tensor(test_data["image"], dtype=torch.float32)
y_test = torch.tensor(test_data["label"])
test_loader = DataLoader(list(zip(X_test, y_test)), batch_size=256)

for X, y in test_loader:
    print(y)

