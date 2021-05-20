import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression

with open("../data/train_hog.pickle", "rb") as f:
    train_data = pickle.load(f)
X = np.array(train_data["hog"])
y = np.array(train_data["label"])
del train_data
print(np.shape(X), len(y))

logistic_reg = LogisticRegression(multi_class="ovr")
logistic_reg.fit(X, y)
print("Training Accuracy:", logistic_reg.score(X, y))

with open("../data/test_hog.pickle", "rb") as f:
    test_data = pickle.load(f)
X_test = np.array(test_data["hog"])
y_test = np.array(test_data["label"])
del test_data
print(np.shape(X_test), len(y_test))

print("Testing  Accuracy:", logistic_reg.score(X_test, y_test))
