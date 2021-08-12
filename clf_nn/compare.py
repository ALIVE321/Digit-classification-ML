from torchvision import models
from model import *
import pickle
import matplotlib.pyplot as plt


if __name__ == "__main__":
    # model = ConvNet_2().to(device)
    model = resnet18
    # print(model)

    with open("../data/train_data.pickle", "rb") as f:
        train_data = pickle.load(f)
    X_train = (torch.tensor(train_data["image"], device=device, dtype=torch.float32).reshape((-1, 1, 32, 32)) - 127) / 255.0
    y_train = torch.tensor(train_data["label"], device=device)
    del train_data
    print(X_train.shape, y_train.shape)
    train_loader = DataLoader(list(zip(X_train, y_train)), batch_size=256, shuffle=True,)
    
    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    
    for epoch in range(50):
        print(f"Epoch {epoch:2d} :")
        train(model, train_loader, loss_func, optimizer)

    with open("../data/test_data.pickle", "rb") as f:
        test_data = pickle.load(f)
    X_test = (torch.tensor(test_data["image"], device=device, dtype=torch.float32).reshape((-1, 1, 32, 32)) - 127) / 255.0
    y_test = torch.tensor(test_data["label"], device=device)
    del test_data
    print(X_test.shape, y_test.shape)
    test_loader = DataLoader(list(zip(X_test, y_test)), batch_size=256)

    test(model, test_loader, loss_func)
    