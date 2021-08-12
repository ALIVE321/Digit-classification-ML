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
    train_indices = np.random.choice(X_train.shape[0], 10000)
    X_train = X_train[train_indices]
    y_train = y_train[train_indices]

    clf_list = []
    for i in range(10):
        print(f"Train {i}-th Logistic Regression Classifier...")
        logistic_reg = KernelLogisticRegression(X_train.shape, 1)
        logistic_reg.optimize_lasso(X_train, (y_train==i).to(torch.float32))
        clf_list.append(logistic_reg)

    with open("../data/test_hog.pickle", "rb") as f:
        test_data = pickle.load(f)
    X_test = torch.tensor(test_data["hog"], device=device)
    y_test = torch.tensor(test_data["label"], device=device)
    del test_data
    print(X_test.shape, y_test.shape)
    test_indices = np.random.choice(X_train.shape[0], 10000)
    X_test = X_test[test_indices]
    y_test = y_test[test_indices]

    predicts = [clf_list[i].predict(X_test) for i in range(10)]

    for i in range(10):
        print(f"{torch.sum((predicts[i] > 0.5) == (y_test == i)).item() / len(y_test):.4f}", end=",")
    print()

    merge_pred = torch.argmax(torch.cat([i.view(-1, 1) for i in predicts], dim=1), dim=1)

    print(torch.sum(merge_pred == y_test).item() / len(y_test))

    torch.save(clf_list, f"./klr_c1.pickle")