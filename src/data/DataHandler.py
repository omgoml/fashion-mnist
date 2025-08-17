import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from config.env import *

def data_handler():
    train_transform = transforms.Compose([
        transforms.RandomRotation(10),
        transforms.RandomAffine(0, translate=(0.1,0.1)),
        transforms.ToTensor(),
        transforms.Normalize((1.307,),(0.3081,))
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((1.307,),(0.3081,))
    ])

    train_data = datasets.FashionMNIST(
        root=ROOT_PATH,
        train=True,
        transform=train_transform, 
        download=True,
    )


    test_data = datasets.FashionMNIST(
        root=ROOT_PATH,
        train=True,
        transform=test_transform, 
        download=True,
    )

    train_loader = DataLoader(
        dataset=train_data,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True if torch.cuda.is_available() else False,
    )


    test_loader = DataLoader(
        dataset=test_data,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    return train_loader, test_loader
