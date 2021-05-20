import numpy as np
from sklearn import datasets
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.preprocessing import PolynomialFeatures
import pickle

# X, y = make_moons( n_samples=100, noise=0.15, random_state=42 )

# def plot_dataset(X, y, axes):
#     plt.plot( X[:,0][y==0], X[:,1][y==0], "bs" )
#     plt.plot( X[:,0][y==1], X[:,1][y==1], "g^" )
#     plt.axis( axes )
#     plt.grid( True, which="both" )
#     plt.xlabel(r"$x_l$")
#     plt.ylabel(r"$x_2$")
#
#
# def plot_predict(clf, axes):
#     x0s = np.linspace(axes[0], axes[1], 100)
#     x1s = np.linspace(axes[2], axes[3], 100)
#     x0, x1 = np.meshgrid( x0s, x1s )
#     X = np.c_[x0.ravel(), x1.ravel()]
#     y_pred = clf.predict( X ).reshape( x0.shape )
#     y_decision = clf.decision_function( X ).reshape( x0.shape )
#     plt.contour( x0, x1, y_pred, cmap=plt.cm.winter, alpha=0.5 )
#     plt.contour( x0, x1, y_decision, cmap=plt.cm.winter, alpha=0.2 )
#
# svm_clf = Pipeline([ ( "scaler", StandardScaler()),
#                      ("svm_clf", SVC(kernel="poly", degree=3, coef0=100, C=0.5, decision_function_shape='ovr'))
#                      ], verbose=True)
# polynomial_svm_clf.fit( X, y )
# plot_dataset( X, y, [-1.5, 2.5, -1, 1.5] )
# plot_predict( polynomial_svm_clf, [-1.5, 2.5, -1, 1.5] )
# plt.show()

with open("../data/train_hog.pickle", "rb") as f:
    train_data = pickle.load(f)
X = np.array(train_data["hog"])
y = np.array(train_data["label"])
del train_data
print(np.shape(X), len(y))

# svm_clf.fit( X, y )
# print(svm_clf.score( X, y ))
std_scaler = StandardScaler()
X_std = std_scaler.fit_transform(X)

svm_seq = []

for i in range(10):
    print(f"Train {i}-th SVM...")
    one2all_y = (y == i)
    # svm_clf = LinearSVC()
    svm_clf = SVC(kernel="poly", degree=3, coef0=100, C=0.5)
    svm_clf.fit(X_std, one2all_y)
    svm_clf.score(X_std, one2all_y)
    svm_seq.append(svm_clf)

