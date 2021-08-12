import numpy as np
from sklearn import datasets
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.preprocessing import PolynomialFeatures
import pickle, sys
import seaborn as sns

with open("../data/train_hog.pickle", "rb") as f:
    train_data = pickle.load(f)
X = np.array(train_data["hog"])
y = np.array(train_data["label"])
del train_data
print("Train Size:", np.shape(X), len(y))

# svm_clf.fit( X, y )
# print(svm_clf.score( X, y ))
std_scaler = StandardScaler()
X_std = std_scaler.fit_transform(X)

# svm_seq = []

# for i in range(10):
#     print(f"Train {i}-th SVM...")
#     one2all_y = (y == i)
#     # svm_clf = LinearSVC(tol=0.01)
#     svm_clf = SVC(kernel="rbf", degree=3, coef0=100, C=0.5, max_iter=1000, tol=0.01)
#     svm_clf.fit(X_std, one2all_y)
#     acc = svm_clf.score(X_std, one2all_y)
#     print(f"Training Acc:", acc)
#     svm_seq.append(svm_clf)

with open(f"./{sys.argv[1]}_svm_seq.pickle", "rb") as f:
    svm_seq = pickle.load(f)

# with open("../data/test_hog.pickle", "rb") as f:
#     test_data = pickle.load(f)
# X_test = np.array(test_data["hog"])
# y_test = np.array(test_data["label"])
# del test_data
# print("Test Size:", np.shape(X_test), len(y_test))

# X_test_std = std_scaler.transform(X_test)

# for i in range(10):
#     print(f"Test {i}-th SVM...")
#     one2all_y = (y_test == i)
#     acc = svm_seq[i].score(X_test_std, one2all_y)
#     print(f"Testinging Acc:", acc)


# with open("./rbf_svm_seq.pickle", "wb") as f:
#     pickle.dump(svm_seq, f)

# visualization
def plt_dist(A, B, num):
    fig = plt.figure(figsize=(10,6))
    ax1 = fig.add_subplot(111)
    ax2 = ax1.twinx()
    plt.title(f'Distribution of Transformed Data in LinearSVM-{i}', fontsize=16)
    ax1.set_xlabel('X', fontsize=16)
    ax1.set_ylabel('P1', fontsize=16)
    sns.histplot(A, bins=100, color="tab:red", ax=ax1)
    sns.histplot(B, bins=100, color="tab:green", ax=ax2)
    ax2.set_ylabel('P2', fontsize=16)
    fig.savefig(f"./svm_linear_{num}.png")
if sys.argv[1] == "linear":
    for i in range(10):
        print(f"Draw {i}-th SVM...")
        X_tran = np.matmul(X_test_std, svm_seq[i].coef_.reshape(-1, ))
        plt_dist(X_tran[y_test == i], X_tran[y_test != i], i)
    exit()

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

def PCA_TSNE(data, n, k):
    print("PCA:")
    pca = PCA(n_components=n)
    tsne = TSNE(n_components=k)
    embeddings = pca.fit_transform(data)
    print("TSNE:")
    res = tsne.fit_transform(embeddings)
    return res

def plt_spv(data, label, sptidx):
    size = data.shape[0]
    fig = plt.figure(figsize=(10,6))
    plt.title(f'2-D Train Samples in SVM-{i}', fontsize=16)
    plt.xlabel('X', fontsize=16)
    plt.ylabel('Y', fontsize=16)
    s = np.ones((size, )) / 500
    c = np.ones((size, ), dtype=object)
    s[label==1] = 0.01
    s[sptidx] = 10
    c[label==1] = "green"
    c[label==0] = 'blue'
    plt.scatter(data[:, 0], data[:, 1], s=s, c=c)
    fig.savefig(f"./svm_{sys.argv[1]}_{i}.png")

# data = PCA_TSNE(X_std, 100, 2)
# with open("./2ddata.pickle", "wb") as f:
#     pickle.dump(data, f)
with open("./2ddata.pickle", "rb") as f:
    data = pickle.load(f)

for i in range(10):
    print(f"Draw {i}-th SVM...")
    one2all_y = (y == i)
    plt_spv(data, one2all_y, svm_seq[i].support_)

