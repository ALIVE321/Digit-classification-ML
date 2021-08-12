from model import *
import pickle
import matplotlib.pyplot as plt
import torchvision.datasets as Dataset
import torchvision.transforms as Transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import argparse


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--model", type=str, default="resnet")
    arg_parser.add_argument("--color", type=int, default=1)
    args = arg_parser.parse_args()

    in_channel = 3
    # if args.model == "resnet":
    #     model = ResNet(in_channel).to(device)
    # elif args.model == "deeper":
    #     model = ResNet_deeper(in_channel).to(device)
    # elif args.model == "convnet":
    #     model = ConvNet(in_channel).to(device)
    # elif args.model == "bottleneck":
    #     model = ConvNet_bottleneck(in_channel).to(device)
    model = torch.load(f"./model_{args.model}_{args.color}.pickle")

    # print(model)
    if args.color:
        train_dset = Dataset.ImageFolder(root="../data/train", transform=Transforms.Compose([
                Transforms.Resize((32, 32)), 
                Transforms.ToTensor(),
                Transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])
        )
    else:
        train_dset = Dataset.ImageFolder(root="../data/train", transform=Transforms.Compose([
                Transforms.Grayscale(num_output_channels=3),
                Transforms.Resize((32, 32)), 
                Transforms.ToTensor(),
                Transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])
        )
    train_dataloader = DataLoader(train_dset, batch_size=256, shuffle=True)

    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

    if args.color:
        test_dset = Dataset.ImageFolder(root="../data/test", transform=Transforms.Compose([
                Transforms.Resize((32, 32)), 
                Transforms.ToTensor(),
                Transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])
        )
    else:
        test_dset = Dataset.ImageFolder(root="../data/test", transform=Transforms.Compose([
                Transforms.Grayscale(num_output_channels=3),
                Transforms.Resize((32, 32)), 
                Transforms.ToTensor(),
                Transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])
        )
    test_dataloader = DataLoader(test_dset, batch_size=256, shuffle=True)

    for epoch in range(20):
        print(f"Epoch {epoch:2d} :")
        train_loss = train(model, train_dataloader, loss_func, optimizer)
        test_loss, acc = test(model, test_dataloader, loss_func)
        with open(f"./log_{args.model}_{args.color}.txt", "a") as f:
            f.write(f"{train_loss} {test_loss} {acc}\n")
    torch.save(model, f"./model_{args.model}_{args.color}.pickle")
