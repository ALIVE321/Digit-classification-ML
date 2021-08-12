import pickle
import sys
from lr_utils import *


if __name__ == "__main__":
    with open("../data/train_hog.pickle", "rb") as f:
        train_data = pickle.load(f)
    X_train = torch.tensor(train_data["hog"], device=device)
    y_train = torch.tensor(train_data["label"], device=device)
    del train_data

    print(X_train.shape, y_train.shape)

    loss_func, param = sys.argv[1], sys.argv[2]
    clf_list = []
    scales = torch.zeros((100, ), device='cpu')
    for i in range(10):
        print(f"Train {i}-th Logistic Regression Classifier...")
        logistic_reg = LogisticRegression(X_train.shape)
        if loss_func == "norm": scales += logistic_reg.optimize(X_train, (y_train==i).to(torch.float32))
        if loss_func == "ridge": scales += logistic_reg.ridge_optimize(X_train, (y_train==i).to(torch.float32), _lambda=float(param))
        if loss_func == "lasso": scales += logistic_reg.lasso_optimize(X_train, (y_train==i).to(torch.float32), _lambda=float(param))
        clf_list.append(logistic_reg)

    loss_type = f"{loss_func}_{param}"
    torch.save(scales, f"scale_{loss_type}.pickle")

    with open("../data/test_hog.pickle", "rb") as f:
        test_data = pickle.load(f)
    X_test = torch.tensor(test_data["hog"], device=device)
    y_test = torch.tensor(test_data["label"], device=device)
    del test_data
    print(X_test.shape, y_test.shape)

    predicts = [clf_list[i].predict(X_test) for i in range(10)]

    for i in range(10):
        print(f"{torch.sum((predicts[i] > 0.5) == (y_test == i)).item() / len(y_test):.4f}", end=",")
    print()

    merge_pred = torch.argmax(torch.cat([i.view(-1, 1) for i in predicts], dim=1), dim=1)

    print(torch.sum(merge_pred == y_test).item() / len(y_test))

    torch.save(clf_list, f"./lr_{loss_type}.pickle")