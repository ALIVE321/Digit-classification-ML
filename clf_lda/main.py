import os
import pickle
import torch
from tqdm import tqdm
import numpy as np
# import matplotlib.pyplot as plt

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
        p = (1/np.sqrt(2*np.pi*self.sm_pos)) * torch.exp((trans_X - self.mu_pos) * (trans_X - self.mu_pos) / (-2 * self.sm_pos))
        return p

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
        clf_lda.fit(X_train, (y_train == i).to(torch.float32))
        clf_list.append(clf_lda)

    with open("../data/test_hog.pickle", "rb") as f:
        test_data = pickle.load(f)
    X_test = torch.tensor(test_data["hog"], device=device)
    y_test = torch.tensor(test_data["label"], device=device)
    del test_data
    print(X_test.shape, y_test.shape)
    
    # plt.scatter(torch.matmul(X_test, clf_list[0]).cpu(), (y_test==0).cpu())
    # plt.savefig("./pic.png")

    predicts = [clf_list[i].predict(X_test).reshape((-1, 1)) for i in range(10)]
    merge_pred = torch.argmax(torch.cat(predicts, dim=1), dim=1)
    print(torch.sum(merge_pred == y_test).item() / len(y_test))

