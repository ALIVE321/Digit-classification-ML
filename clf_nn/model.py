import torch
import numpy as np
import tqdm
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_layer_seq = nn.Sequential(
            nn.Linear(32*32*3, 1024),
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


class ConvNet(nn.Module):
    def __init__(self, in_channel):
        super(ConvNet, self).__init__()
        self.conv1 = self.construct_block(in_channel, 32, 3, 1, 1)
        self.conv2 = self.construct_block(32, 32, 3, 1, 1)
        self.pool1 = nn.MaxPool2d(2, 2, 1)
        self.conv3 = self.construct_block(32, 64, 3, 1, 1)
        self.conv4 = self.construct_block(64, 64, 3, 1, 1)
        self.pool2 = nn.MaxPool2d(2, 2, 1)
        self.conv5 = self.construct_block(64, 128, 3, 1, 1)
        self.conv6 = self.construct_block(128, 128, 3, 1, 1)
        self.pool3 = nn.MaxPool2d(2, 2, 1)
        self.conv7 = self.construct_block(128, 256, 3, 1, 1)
        self.pool4 = nn.MaxPool2d(2, 2, 1)
        self.conv8 = self.construct_block(256, 256, 3, 1, 1)
        self.pool5 = nn.MaxPool2d(2, 2, 1)
        self.fc1 = nn.Sequential(nn.Linear(256*2*2, 1024), nn.ReLU())
        self.fc2 = nn.Linear(1024, 10)
        self.softmax = nn.Softmax(dim=1)

    def construct_block(self, in_channel, kernel_num, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(in_channel, kernel_num, kernel_size, stride, padding),
            nn.BatchNorm2d(kernel_num),
            nn.ReLU()
        )

    def forward(self, X):
        X = self.conv1(X)
        X = self.conv2(X)
        X = self.pool1(X)
        X = self.conv3(X)
        X = self.conv4(X)
        X = self.pool2(X)
        X = self.conv5(X)
        X = self.conv6(X)
        X = self.pool3(X)
        X = self.conv7(X)
        X = self.pool4(X)
        X = self.conv8(X)
        X = self.pool5(X)
        X = self.fc1(X.view(X.shape[0], -1))
        X = self.fc2(X)
        p = self.softmax(X)
        return p        


class ConvNet_bottleneck(nn.Module):
    def __init__(self, in_channel):
        super(ConvNet_bottleneck, self).__init__()
        self.conv1 = self.construct_block(in_channel, 32, 3, 1, 1)
        self.conv2 = self.construct_block(32, 32, 3, 1, 1)
        self.pool1 = nn.MaxPool2d(2, 2, 1)
        self.conv3 = self.construct_block(32, 64, 3, 1, 1)
        self.conv4 = self.construct_block(64, 64, 3, 1, 1)
        self.pool2 = nn.MaxPool2d(2, 2, 1)
        self.conv5 = self.construct_block(64, 128, 3, 1, 1)
        self.conv6 = self.construct_block(128, 128, 3, 1, 1)
        self.pool3 = nn.MaxPool2d(2, 2, 1)
        self.conv7 = self.construct_block(128, 1, 3, 1, 1)
        self.pool4 = nn.MaxPool2d(2, 2, 1)
        self.conv_bn = self.construct_block(1, 1, 3, 1, 1)
        self.conv8 = self.construct_block(1, 256, 3, 1, 1)
        self.pool5 = nn.MaxPool2d(2, 2, 1)
        self.fc1 = nn.Sequential(nn.Linear(256*2*2, 1024), nn.ReLU())
        self.fc2 = nn.Linear(1024, 10)
        self.softmax = nn.Softmax(dim=1)

    def construct_block(self, in_channel, kernel_num, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(in_channel, kernel_num, kernel_size, stride, padding),
            nn.BatchNorm2d(kernel_num),
            nn.ReLU()
        )

    def forward(self, X):
        X = self.conv1(X)
        X = self.conv2(X)
        X = self.pool1(X)
        X = self.conv3(X)
        X = self.conv4(X)
        X = self.pool2(X)
        X = self.conv5(X)
        X = self.conv6(X)
        X = self.pool3(X)
        X = self.conv7(X)
        X = self.pool4(X)
        X = self.conv_bn(X)
        X = self.conv8(X)
        X = self.pool5(X)
        X = self.fc1(X.view(X.shape[0], -1))
        X = self.fc2(X)
        p = self.softmax(X)
        return p       


class BasicBlock(nn.Module):
    def __init__(self, in_channel, out_channel, stride, padding=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 3, stride, padding),
            nn.BatchNorm2d(out_channel),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, 3, 1, padding),
            nn.BatchNorm2d(out_channel)
        )
        if stride != 1 or in_channel != out_channel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, 1, stride),
                nn.BatchNorm2d(out_channel)
            )
        else:
            self.shortcut = nn.Sequential()
    
    def forward(self, X):
        out = self.conv1(X)
        out = self.conv2(out)
        out = self.shortcut(X) + out
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, in_channel):
        super(ResNet, self).__init__()
        self.conv0 = nn.Sequential(
            nn.Conv2d(in_channel, 64, 3, 2, 3),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.conv1 = nn.Sequential(
            BasicBlock(64, 64, 2),
            BasicBlock(64, 64, 1)
        )
        self.conv2 = nn.Sequential(
            BasicBlock(64, 128, 2),
            BasicBlock(128, 128, 1)
        )
        self.conv3 = nn.Sequential(
            BasicBlock(128, 256, 2),
            BasicBlock(256, 256, 1)
        )
        self.avgpool = nn.AvgPool2d(2)
        self.fc = nn.Linear(256, 10)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, X):
        out = self.conv0(X)
        out = self.conv1(out)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.avgpool(out)
        out = out.view(out.shape[0], -1)
        logits = self.fc(out)
        p = self.softmax(logits)
        return p


class ResNet_deep(nn.Module):
    def __init__(self, in_channel):
        super(ResNet_deep, self).__init__()
        self.conv0 = nn.Sequential(
            nn.Conv2d(in_channel, 64, 3, 2, 3),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.conv11 = nn.Sequential(
            BasicBlock(64, 32, 2),
            BasicBlock(32, 32, 1)
        )
        self.conv12 = nn.Sequential(
            BasicBlock(32, 32, 1),
            BasicBlock(32, 64, 1)
        )
        self.conv21 = nn.Sequential(
            BasicBlock(64, 64, 2),
            BasicBlock(64, 64, 1)
        )
        self.conv22 = nn.Sequential(
            BasicBlock(64, 64, 1),
            BasicBlock(64, 128, 1)
        )
        self.conv31 = nn.Sequential(
            BasicBlock(128, 128, 2),
            BasicBlock(128, 128, 1)
        )
        self.conv32 = nn.Sequential(
            BasicBlock(128, 128, 1),
            BasicBlock(128, 256, 1)
        )
        self.avgpool = nn.AvgPool2d(2)
        self.fc = nn.Linear(256, 10)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, X):
        out = self.conv0(X)
        out = self.conv11(out)
        out = self.conv12(out)
        out = self.conv21(out)
        out = self.conv22(out)
        out = self.conv31(out)
        out = self.conv32(out)
        out = self.avgpool(out)
        out = out.view(out.shape[0], -1)
        logits = self.fc(out)
        p = self.softmax(logits)
        return p

class ResNet_deeper(nn.Module):
    def __init__(self, in_channel):
        super(ResNet_deeper, self).__init__()
        self.conv0 = nn.Sequential(
            nn.Conv2d(in_channel, 64, 3, 2, 3),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.conv11 = nn.Sequential(
            BasicBlock(64, 64, 2),
            BasicBlock(64, 64, 1)
        )
        self.conv12 = nn.Sequential(
            BasicBlock(64, 64, 1),
            BasicBlock(64, 64, 1)
        )
        self.conv21 = nn.Sequential(
            BasicBlock(64, 128, 2),
            BasicBlock(128, 128, 1)
        )
        self.conv22 = nn.Sequential(
            BasicBlock(128, 128, 1),
            BasicBlock(128, 256, 1)
        )
        self.conv31 = nn.Sequential(
            BasicBlock(256, 256, 2),
            BasicBlock(256, 256, 1)
        )
        self.conv32 = nn.Sequential(
            BasicBlock(256, 256, 1),
            BasicBlock(256, 256, 1)
        )
        self.avgpool = nn.AvgPool2d(2)
        self.fc = nn.Linear(256, 10)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, X):
        out = self.conv0(X)
        out = self.conv11(out)
        out = self.conv12(out)
        out = self.conv21(out)
        out = self.conv22(out)
        out = self.conv31(out)
        out = self.conv32(out)
        out = self.avgpool(out)
        out = out.view(out.shape[0], -1)
        logits = self.fc(out)
        p = self.softmax(logits)
        return p


def train(model, data_loader, loss_func, optimizer):
    model.train()
    size = len(data_loader.dataset)
    running_loss, running_acc = 0, 0
    for batch, (X, y) in enumerate(data_loader):
        X = X.to(device=device)
        y = y.to(device=device)
        pred = model(X)
        loss = loss_func(pred, y)
        running_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batch % 100 == 0:
            acc = (pred.argmax(1) == y).type(torch.float).sum().item() / len(X) * 100
            print(f"\rLoss: {running_loss:>.7f}, Acc: {acc:>.3f}%, Trained: [{(batch+1)*len(X):>5d} / {size:>5d}]", end="")
    return running_loss / size


def test(model, data_loader, loss_func):
    model.eval()
    size = len(data_loader.dataset)
    correct, loss = 0, 0
    with torch.no_grad():
        for X, y in data_loader:
            X = X.to(device=device)
            y = y.to(device=device)
            pred = model(X)
            loss += loss_func(pred, y)
            correct += (pred.argmax(1)==y).type(torch.float).sum().item()
    print(f"Avg Loss: {loss.item()/size:>.7f}, Acc: {correct/size*100:>.3f}%, Test on: {size:>5d}")
    return loss.item() / size, correct / size
