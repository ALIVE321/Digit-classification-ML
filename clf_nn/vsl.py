import os
import cv2
import pickle
from tqdm import tqdm
import numpy as np
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
sys.path.append(".")
from model import *
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import seaborn as sns
import torchvision.datasets as Dataset
import torchvision.transforms as Transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image

model_name = sys.argv[1]
color = int(sys.argv[2])
seed = 123233
np.random.seed(seed)
torch.manual_seed(seed)
import random
random.seed(seed)

def PCA_TSNE(data, n, k):
    print("PCA:")
    pca = PCA(n_components=n)
    tsne = TSNE(n_components=k)
    embeddings = pca.fit_transform(data)
    print("TSNE:")
    res = tsne.fit_transform(embeddings)
    return res

def plt_fig(data, label):
    size = data.shape[0]
    fig = plt.figure(figsize=(10,6))
    plt.title(f'2-D Test Samples in {sys.argv[3]}', fontsize=16)
    plt.xlabel('X', fontsize=16)
    plt.ylabel('Y', fontsize=16)
    # for i in range(10):
    sns.scatterplot(x=data[:, 0], y=data[:, 1], size=1e-7, hue=["C-%d" % i for i in label], alpha=0.5)
    fig.savefig(f"./d_{model_name}_{color}.png")

def plt_curve():
    fig = plt.figure(figsize=(10,6))
    plt.title(f'Curves of {sys.argv[3]}', fontsize=16)
    ax1 = fig.add_subplot(111)
    plt.ylim([0.005, 0.008])
    ax2 = ax1.twinx()
    ax1.set_xlabel('Epochs', fontsize=16)
    ax1.set_ylabel('Loss', fontsize=16)
    ax2.set_ylabel('Accuracy', fontsize=16)
    plt.ylim([0.8, 1.0])
    train_loss, test_loss, acc = [], [], []
    with open(f"./log_{model_name}_{color}.txt", "r") as f:
        for line in f.readlines():
            a, b, c = [float(i) for i in line.strip().split()]
            train_loss.append(a)
            test_loss.append(b)
            acc.append(c)
    sns.lineplot(range(30), train_loss, ax=ax1, color="tab:red", label="Train Loss")
    sns.lineplot(range(30), test_loss, ax=ax1, color="tab:blue", label="Test Loss")
    sns.lineplot(range(30), acc, ax=ax2, color="tab:green")
    plt.legend()
    fig.savefig(f"./model_{model_name}_{color}.png")

# plt_curve()

model = torch.load(f"./model_{model_name}_{color}.pickle").cpu()
train_dset = Dataset.ImageFolder(root="../data/train", transform=Transforms.Compose([
        Transforms.Grayscale(num_output_channels=3),
        Transforms.Resize((32, 32)), 
        Transforms.ToTensor(),
        Transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
)
sample_size = 10000
train_dataloader = DataLoader(train_dset, batch_size=sample_size, shuffle=True)

X, y = next(iter(train_dataloader))
y = np.array(y, dtype=int)
print(y)

with torch.no_grad():
    for module_pos, module in model._modules.items():
        print(module_pos)
        X = module(X)
        if module_pos == "conv3":
            F = X.view(sample_size, -1).numpy()
            break

F2 = PCA_TSNE(F, 100, 2)

print(F2.shape, y.shape)
plt_fig(F2, y)
    
