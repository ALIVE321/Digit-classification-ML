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

fig = plt.figure(figsize=(10,6))
plt.plot(torch.load("./clf_lr/scale_norm_0.pickle").cpu(), label="norm")
plt.plot(torch.load("./clf_lr/scale_ridge_1e-1.pickle").cpu(), label="ridge 1e-1")
plt.plot(torch.load("./clf_lr/scale_lasso_1e-1.pickle").cpu(), label="lasso 1e-1")
plt.plot(torch.load("./clf_lr/scale_ridge_1e-2.pickle").cpu(), label="ridge 1e-2")
plt.plot(torch.load("./clf_lr/scale_lasso_1e-2.pickle").cpu(), label="lasso 1e-2")
plt.plot(torch.load("./clf_lr/scale_ridge_1e-3.pickle").cpu(), label="ridge 1e-3")
plt.plot(torch.load("./clf_lr/scale_lasso_1e-3.pickle").cpu(), label="lasso 1e-3")
plt.title(r"$\beta^2$ - Epochs")
plt.xlabel("Epochs")
plt.ylabel(r"$\beta^2$")
plt.legend()
plt.savefig("png/scales.png")
exit()

# lda_clfs = torch.load("./clf_lda/lda.pickle")
lrn_clfs = torch.load("./clf_lr/lr_norm_0.pickle")
lrl1_clfs = torch.load("./clf_lr/lr_lasso_1e-1.pickle")
lrr1_clfs = torch.load("./clf_lr/lr_ridge_1e-1.pickle")
lrl2_clfs = torch.load("./clf_lr/lr_lasso_1e-2.pickle")
lrr2_clfs = torch.load("./clf_lr/lr_ridge_1e-2.pickle")
lrl3_clfs = torch.load("./clf_lr/lr_lasso_1e-3.pickle")
lrr3_clfs = torch.load("./clf_lr/lr_ridge_1e-3.pickle")

X = range(2916)

for i in range(10):
    fig = plt.figure(figsize=(10,6))
    sns.distplot(lrn_clfs[i].beta.cpu(), bins=100, label="Normal LR", hist=0)
    sns.distplot(lrl1_clfs[i].beta.cpu(), bins=100, label="LASSO-1e-1 LR", hist=0)
    sns.distplot(lrr1_clfs[i].beta.cpu(), bins=100, label="Ridge-1e-1 LR", hist=0)
    sns.distplot(lrl2_clfs[i].beta.cpu(), bins=100, label="LASSO-1e-2 LR", hist=0)
    sns.distplot(lrr2_clfs[i].beta.cpu(), bins=100, label="Ridge-1e-2 LR", hist=0)
    sns.distplot(lrl3_clfs[i].beta.cpu(), bins=100, label="LASSO-1e-3 LR", hist=0)
    sns.distplot(lrr3_clfs[i].beta.cpu(), bins=100, label="Ridge-1e-3 LR", hist=0)
    plt.legend()
    plt.savefig(f"png/beta_clf-{i}.png")

