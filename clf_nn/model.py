import torch
import numpy as np
import tqdm
import torch.nn as nn
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


class SimpleNet(nn.Module):
    def __init__(self):
        self.input_size = (32, 32)
        super(SimpleNet, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_layer_seq = nn.Sequential(
            nn.Linear(np.prod(self.input_size), 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, X):
        X = self.flatten(X)
        logits = self.linear_layer_seq(X)
        p = self.softmax(logits)
        return p


def train(model, data_loader, loss_func, optimizer):
    size = len(data_loader.dataset)
    for batch, (X, y) in enumerate(data_loader):
        pred = model(X)
        loss = loss_func(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batch % 100 == 0:
            print(f"Loss: {loss.item():>.7f}, Acc: {(pred.argmax(1)==y).type(torch.float).sum().item()/len(X)*100:>.3f}%, Trained: [{(batch+1)*len(X):>5d} / {size:>5d}]")


def test(model, data_loader, loss_func):
    size = len(data_loader.dataset)
    correct, loss = 0, 0
    with torch.no_grad():
        for X, y in enumerate(data_loader):
            pred = model(X)
            loss += loss_func(pred, y)
            correct += (pred.argmax(1)==y).type(torch.float).sum().item()
    print(f"Loss: {loss.item():>4.7f}, Acc: {correct/len(X)*100:>.3f}%, Test on: {size:>5d}")
