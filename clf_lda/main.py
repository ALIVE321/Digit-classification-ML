import os
import pickle
import torch
from tqdm import tqdm
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


class LDA:
    def __init__(self):
        self.beta = None
        self.mu_pos = None
        self.mu_neg = None
        self.sm_pos = None
        self.sm_neg = None
    
    def fit(self, X, y):
        X_pos, X_neg = X[y == 1], X[y == 0]
        dims = X.shape[1]
        self.mu_pos, self.mu_neg = X_pos.mean(axis=0), X_neg.mean(axis=0)
        res_pos, res_neg = X_pos - self.mu_pos, X_neg - self.mu_neg
        var_within = torch.matmul(res_pos.T, res_pos) + torch.matmul(res_neg.T, res_neg)
        self.beta = torch.matmul(torch.inverse(var_within), self.mu_pos - self.mu_neg)
        trans_X_pos, trans_X_neg = torch.matmul(X_pos, self.beta), torch.matmul(X_neg, self.beta)
        self.mu_pos, self.mu_neg = trans_X_pos.mean().item(), trans_X_neg.mean().item()
        self.sm_pos = (trans_X_pos - self.mu_pos).dot(trans_X_pos - self.mu_pos).item() / X_pos.shape[0]
        self.sm_neg = (trans_X_neg - self.mu_neg).dot(trans_X_neg - self.mu_neg).item() / X_neg.shape[0]

    def predict(self, X):
        trans_X = torch.matmul(X, self.beta)
        p1 = (1/np.sqrt(2*np.pi*self.sm_pos)) * torch.exp((trans_X - self.mu_pos) * (trans_X - self.mu_pos) / (-2 * self.sm_pos))
        p2 = (1/np.sqrt(2*np.pi*self.sm_neg)) * torch.exp((trans_X - self.mu_neg) * (trans_X - self.mu_neg) / (-2 * self.sm_neg))
        return p1 / (p1 + p2)


def plt_dist(A, B, num):
    fig = plt.figure(figsize=(10,6))
    ax1 = fig.add_subplot(111)
    ax2 = ax1.twinx()
    plt.title(f'Distribution of Transformed Data in CLF-{i}', fontsize=16)
    ax1.set_xlabel('X', fontsize=16)
    ax1.set_ylabel('P1', fontsize=16)
    sns.histplot(A, bins=100, color="tab:red", ax=ax1)
    sns.histplot(B, bins=100, color="tab:green", ax=ax2)
    ax2.set_ylabel('P2', fontsize=16)
    fig.savefig(f"./lda_clf_{num}.png")


if __name__ == "__main__":
    with open("../data/train_hog.pickle", "rb") as f:
        train_data = pickle.load(f)
    X_train = torch.tensor(train_data["hog"], device=device)
    y_train = torch.tensor(train_data["label"], device=device)
    del train_data
    print(X_train.shape, y_train.shape)

    clf_list = []
    for i in range(10):
        print(f"Train {i}-th LDA Classifier...")
        clf_lda = LDA()
        pos_X = X_train[(y_train == i)]
        pos_size = pos_X.shape[0]
        X_pad = torch.zeros((pos_size*8 + X_train.shape[0], X_train.shape[1]), device=device, dtype=torch.float)
        y_pad = torch.ones((pos_size*8 + y_train.shape[0]), device=device, dtype=torch.float)
        y_pad[: y_train.shape[0]] = (y_train == i)
        X_pad[: X_train.shape[0]] = X_train
        print(X_pad.shape, y_pad.shape)
        for _ in range(8):
            X_pad[X_train.shape[0]+pos_size*_: X_train.shape[0]+pos_size*(_+1)] = pos_X
        clf_lda.fit(X_pad, y_pad)
        # clf_lda.fit(X_train, (y_train == i).to(torch.float32))
        clf_list.append(clf_lda)
        X_trans = torch.matmul(X_train, clf_lda.beta)
        # plt_dist(X_trans[y_train==i].cpu(), X_trans[y_train!=i].cpu(), i)

    torch.save(clf_list, "./lda.pickle")

    with open("../data/test_hog.pickle", "rb") as f:
        test_data = pickle.load(f)
    X_test = torch.tensor(test_data["hog"], device=device)
    y_test = torch.tensor(test_data["label"], device=device)
    del test_data
    print(X_test.shape, y_test.shape)

    predicts = [clf_list[i].predict(X_test).reshape((-1, 1)) for i in range(10)]
    merge_pred = torch.argmax(torch.cat(predicts, dim=1), dim=1)
    print(torch.sum(merge_pred == y_test).item() / len(y_test))

