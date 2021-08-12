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
from sklearn.decomposition import PCA
from scipy.signal import savgol_filter

sys.path.append("./clf_lda")
sys.path.append("./clf_lr")

lda_list = torch.load("./clf_lda/lda.pickle")
lr_list = torch.load("./clf_lr/lr_norm_0.pickle")


with open("./data/train_hog.pickle", "rb") as f:
    test_data = pickle.load(f)
X_test = np.array(test_data["hog"])
y_test = np.array(test_data["label"])
del test_data
print(X_test.shape, y_test.shape)

pca = PCA(n_components=1)
pca.fit(X_test)
for i in range(10):
    print(pca.components_[0].dot(lda_list[i].beta.cpu().numpy()), end=" & ")
    print(pca.components_[0].dot(lr_list[i].beta.cpu().numpy()), end=" & ")
    print(lda_list[i].beta.cpu().numpy().dot(lr_list[i].beta.cpu().numpy()), end="\\\\\n")
    

