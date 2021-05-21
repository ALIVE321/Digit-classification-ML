from model import *
import pickle
import matplotlib.pyplot as plt


if __name__ == "__main__":
    model = SimpleNet().to(device)
    print(model)

    with open("../data/train_data.pickle", "rb") as f:
        train_data = pickle.load(f)
    X_train = torch.tensor(train_data["image"], device=device, dtype=torch.float32)
    y_train = torch.tensor(train_data["label"], device=device)
    del train_data
    print(X_train.shape, y_train.shape)

    train_loader = DataLoader(list(zip(X_train, y_train)), batch_size=128, shuffle=True,)

    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    train(model, train_loader, loss_func, optimizer)

    with open("../data/test_data.pickle", "rb") as f:
        test_data = pickle.load(f)
    X_test = torch.tensor(test_data["image"], device=device, dtype=torch.float32)
    y_test = torch.tensor(test_data["label"], device=device)
    del test_data
    print(X_test.shape, y_test.shape)

    test_loader = DataLoader(list(zip(X_test, y_test)), batch_size=256)

    test(model, test_loader, loss_func)
    