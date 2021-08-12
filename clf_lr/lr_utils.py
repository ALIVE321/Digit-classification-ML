import numpy as np
import torch
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

def sigmoid(x):
    return 1 / (1 + torch.exp(-1 * x))


class LogisticRegression:
    def __init__(self, shape):
        self.n, self.p = shape
        self.beta = torch.zeros((self.p,), device=device)
        self.best_beta = None
        self.best_acc = 0

    def predict(self, X):
        return sigmoid(torch.matmul(X, self.beta))

    def optimize(self, X, y, lr=0.001, maxIter=100, batchSize=1024):
        itr = 0
        scale = []
        while itr < maxIter:
            for i in range(self.n // batchSize + 1):
                indices = np.random.choice(self.n, batchSize)
                X_batch = X[indices]
                y_batch = y[indices]
                p = self.predict(X_batch)
                partial_beta = torch.sum((y_batch - p).reshape((-1, 1)) * X_batch, dim=0)
                self.beta = self.beta + partial_beta * lr
                p = self.predict(X_batch)
                acc = torch.sum((p > 0.5) == y_batch).item() / len(p)
                if acc > self.best_acc:
                    self.best_acc = acc
                    self.best_beta = self.beta
                print(f"\rIteration: {itr:4d}, Epoch: {i:3d}, Training Acc: {acc:.4f}", end="")
            itr += 1
            scale.append(self.beta.dot(self.beta).item())
        self.beta = self.best_beta
        print(f"\rIteration: {itr:4d}, Best training Acc: {self.best_acc:.4f}")
        return torch.tensor(scale, device='cpu')


    def ridge_optimize(self, X, y, lr=0.001, maxIter=100, batchSize=1000, _lambda=0.001):
        itr = 0
        scale = []
        while itr < maxIter:
            for i in range(self.n // batchSize + 1):
                indices = np.random.choice(self.n, batchSize)
                X_batch = X[indices]
                y_batch = y[indices]
                p = self.predict(X_batch)
                partial_beta = torch.sum((y_batch - p).reshape((-1,1)) * X_batch, dim=0) - _lambda * self.beta
                self.beta = self.beta + partial_beta * lr
                p = self.predict(X_batch)
                acc = torch.sum((p > 0.5) == y_batch).item() / len(p)
                if acc > self.best_acc:
                    self.best_acc = acc
                    self.best_beta = self.beta
                print(f"\rIteration: {itr:4d}, Epoch: {i:3d}, Training Acc: {acc:.4f}", end="")
            itr += 1
            scale.append(self.beta.dot(self.beta).item())
        self.beta = self.best_beta
        print(f"\rIteration: {itr:4d}, Best training Acc: {self.best_acc:.4f}")
        return torch.tensor(scale, device='cpu')

    # def lasso_optimize(self, X, y, _lambda=10, maxIter=5):
    #     dims = X.shape[1]
    #     # X_norm = torch.sum(X * X).item()
    #     for itr in range(maxIter):
    #         for d in tqdm(range(dims), ncols=80):
    #             indices = [i for i in range(dims) if i != d]
    #             residual = y - torch.matmul(X[:, indices], self.beta[indices])
    #             norm = torch.dot(X[:, d], X[:, d]).item()
    #             y_est = torch.dot(residual, X[:, d]).item() / norm
    #             self.beta[d] = np.sign(y_est) * max(.0, abs(y_est) - _lambda / norm)
    #         # print(self.beta)
    #         _lambda /= 10
    #         p = self.predict(X)
    #         acc = torch.sum((p > 0.5) == y).item() / len(p)
    #         print(f"Iteration: {itr:4d}, Training Acc: {acc:.4f}")

    def lasso_optimize(self, X, y, lr=0.001, maxIter=100, batchSize=1000, _lambda=0.1):
        itr = 0
        scale = []
        while itr < maxIter:
            for i in range(self.n // batchSize + 1):
                indices = np.random.choice(self.n, batchSize)
                X_batch = X[indices]
                y_batch = y[indices]
                p = self.predict(X_batch)
                partial_beta = torch.sum((y_batch - p).reshape((-1,1)) * X_batch, dim=0) - _lambda * torch.sign(self.beta)
                self.beta = self.beta + partial_beta * lr
                p = self.predict(X_batch)
                acc = torch.sum((p > 0.5) == y_batch).item() / len(p)
                if acc > self.best_acc:
                    self.best_acc = acc
                    self.best_beta = self.beta
                print(f"\rIteration: {itr:4d}, Epoch: {i:3d}, Training Acc: {acc:.4f}", end="")
            itr += 1
            scale.append(self.beta.dot(self.beta).item())
        self.beta = self.best_beta
        print(f"\rIteration: {itr:4d}, Best training Acc: {self.best_acc:.4f}")
        return torch.tensor(scale, device='cpu')


class KernelLogisticRegression:
    def __init__(self, shape, coef=1):
        self.n, self.p = shape[:2]
        self.c = torch.zeros((self.n,), device=device)
        self.X = None
        self.coef = coef
        self.best_acc = 0
        self.best_c = 0
    
    def kernel_func(self, x):
        k = torch.zeros((self.n,), device=device)
        x = x.view(1, self.p)
        diff = (x - self.X)
        rbf = torch.exp(torch.sum(diff * diff, dim=1) / (-2 * self.coef * self.coef))
        return rbf

    def predict(self, X):
        return sigmoid(torch.tensor([self.c.dot(self.kernel_func(x)).item() for x in X], device=device))
    
    def optimize_lasso(self, X, y, lr=0.001, maxIter=100, batchSize=1000, _lambda=0.01):
        self.X = X
        tmp_K = torch.cat([self.kernel_func(x).view(1, -1) for x in X], dim=0)
        # print(tmp_K.shape)
        itr = 0
        while itr < maxIter:
            for i in range(self.n // batchSize + 1):
                indices = np.random.choice(self.n, batchSize)
                y_batch = y[indices]
                p = sigmoid(torch.matmul(tmp_K[indices], self.c.view(-1, 1)).view(-1))
                partial_c = torch.sum((y_batch - p).reshape((-1, 1)) * tmp_K[indices], dim=0) - _lambda * torch.sign(self.c)
                self.c = self.c + partial_c * lr
                p = sigmoid(torch.matmul(tmp_K[indices], self.c.view(-1, 1)).view(-1))
                acc = torch.sum((p > 0.5) == y_batch).item() / len(p)
                if acc > self.best_acc:
                    self.best_acc = acc
                    self.best_c = self.c
                print(f"\rIteration: {itr:4d}, Epoch: {i:3d}, Training Acc: {acc:.4f}", end="")
            itr += 1
        self.c = self.best_c
        print(f"\rIteration: {itr:4d}, Best training Acc: {self.best_acc:.4f}")