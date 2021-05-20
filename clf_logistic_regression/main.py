import pickle
from lr_utils import *


if __name__ == "__main__":
    with open("../data/train_hog.pickle", "rb") as f:
        train_data = pickle.load(f)
    X_train = torch.tensor(train_data["hog"], device="cuda")
    y_train = torch.tensor(train_data["label"])
    del train_data

    print(X_train.shape, y_train.shape)

    clf_list = []

    for i in range(10):
        print(f"Train {i}-th Logistic Regression Classifier...")
        logistic_reg = LogisticRegression(X_train.shape)
        # logistic_reg.optimize(X_train, (y_train==i).to(torch.float32), lr=0.001, maxIter=100)
        # logistic_reg.ridge_optimize(X_train, (y_train==i).to(torch.float32), lr=0.001, maxIter=100, _lambda=.001)
        logistic_reg.lasso_optimize(X_train, (y_train==i).to(torch.float32), maxIter=100, _lambda=.001)
        clf_list.append(logistic_reg)

    with open("../data/test_hog.pickle", "rb") as f:
        test_data = pickle.load(f)
    X_test = torch.tensor(test_data["hog"])
    y_test = torch.tensor(test_data["label"])
    del test_data
    print(X_test.shape, y_test.shape)

    predicts = [clf_list[i].predict(X_test).reshape((-1, 1)) for i in range(10)]

    merge_pred = torch.argmax(torch.cat(predicts, dim=1), dim=1)

    print((torch.sum(merge_pred == y_test) / len(y_test)).item())

    with open("./lr_lasso.pickle", "wb") as f:
    # with open("./lr_ridge.pickle", "wb") as f:
        pickle.dump(clf_list, f)