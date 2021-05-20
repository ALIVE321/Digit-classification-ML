import os
import pickle
import torch
from tqdm import tqdm

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


def LDA(X, y):
    X_pos, X_neg = X[y == 1], X[y == 0]
    dims = X.shape[1]
    print(dims)
    mu_pos, mu_neg = X_pos.mean(axis=0), X_neg.mean(axis=0)
    res_pos, res_neg = X_pos - mu_pos, X_neg - mu_neg
    var_within = torch.zeros((dims, dims), device=device)
    for xi in tqdm(res_pos, ncols=80):
        var_within += torch.matmul(xi.reshape(-1, 1), xi.reshape(1, -1))
    for xi in tqdm(res_neg, ncols=80):
        var_within += torch.matmul(xi.reshape(-1, 1), xi.reshape(1, -1))
    beta = torch.matmul(torch.inverse(var_within), mu_pos - mu_neg)
    return beta


if __name__ == "__main__":
    with open("../data/train_hog.pickle", "rb") as f:
        train_data = pickle.load(f)
    X_train = torch.tensor(train_data["hog"], device=device)
    y_train = torch.tensor(train_data["label"], device=device)
    del train_data
    print(X_train.shape, y_train.shape)

    clf_list = []
    for i in range(1):
        print(f"Train {i}-th Logistic Regression Classifier...")
        clf_list.append(LDA(X_train, (y_train == i).to(torch.float32)))

    with open("../data/test_hog.pickle", "rb") as f:
        test_data = pickle.load(f)
    X_test = torch.tensor(test_data["hog"], device=device)
    y_test = torch.tensor(test_data["label"], device=device)
    del test_data
    print(X_test.shape, y_test.shape)

    from matplotlib import pyplot as plt
    plt.plot(torch.matmul(X_test, clf_list[0]), y_test==0)
    plt.save("./pic.png")
