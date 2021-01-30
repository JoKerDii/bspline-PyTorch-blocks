# test

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch import optim
from torch.utils.data import Dataset
from torchvision import transforms

from BSplineActivation import BSplineActivation


class FashionMNIST(Dataset):

    """
    Dataset from Kaggle competition
    """

    def __init__(self):
        self.transform = transforms.Compose([transforms.ToTensor()])
        fashion_df = pd.read_csv(
            '/home/zhendi/wei/splines-nn/fashion-mnist_train.csv')
        self.labels = fashion_df.label.values
        self.images = fashion_df.iloc[:, 1:].values.astype(
            'uint8').reshape(-1, 28, 28)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        label = self.labels[idx]
        img = Image.fromarray(self.images[idx])

        if self.transform:
            img = self.transform(img)

        return img, label


def train_mlp_model(model, config):
    """
    Function trains the model and prints out the training loss.
    """

    criterion = nn.NLLLoss().to(config.device)
    learning_rate = 0.003
    epochs = 5
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for e in range(epochs):
        running_loss = 0
        for images, labels in trainloader:
            images = images.view(images.shape[0], -1).to(config.device)
            labels = labels.to(config.device)
            log_ps = model(images)
            loss = criterion(log_ps, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        else:
            print(f"Training loss: {running_loss}")


def train_cnn_model(model, config):
    '''
    Function trains the model and prints out the training loss.
    '''
    criterion = nn.NLLLoss().to(config.device)
    learning_rate = 0.003
    epochs = 5
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for e in range(epochs):
        running_loss = 0
        for images, labels in trainloader:
            # images = images.view(images.shape[0], -1)
            images = images.to(config.device)
            labels = labels.to(config.device)
            log_ps = model(images)
            loss = criterion(log_ps, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        else:

            print(f"Training loss: {running_loss}")


class MLP(nn.Module):
    '''
    Simple fully-connected classifier model to demonstrate activation.
    '''

    def __init__(self, config):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(784, 256)  # (N, 28 * 28) -> (N, 256)
        self.fc2 = nn.Linear(256, 128)  # -> (N, 128)
        self.fc3 = nn.Linear(128, 64)  # -> (N, 64)
        self.fc4 = nn.Linear(64, 10)  # -> (N, 10)
        self.a1 = BSplineActivation(num_activations=256,
                                    mode='linear', device=config.device)
        self.a2 = BSplineActivation(num_activations=128,
                                    mode='linear', device=config.device)
        self.a3 = BSplineActivation(num_activations=64,
                                    mode='linear', device=config.device)

    def forward(self, x):

        x = x.view(x.shape[0], -1)
        x = self.a1(self.fc1(x))
        x = self.a2(self.fc2(x))
        x = self.a3(self.fc3(x))
        x = F.log_softmax(self.fc4(x), dim=1)

        return x


class CNN(nn.Module):
    '''
    Simple fully-connected classifier model to demonstrate activation.
    '''

    def __init__(self, config):
        super(CNN, self).__init__()

        self.c1 = 6

        self.conv1 = nn.Conv2d(1, self.c1, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(self.c1 * 12 * 12, 512)  # 864
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 10)
        self.a1 = BSplineActivation(
            num_activations=self.c1, device=config.device)
        self.a2 = torch.nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.a1(x)
        x = self.pool(x)
        x = x.view(-1, self.c1 * 12 * 12)
        x = self.a2(self.fc1(x))
        x = self.a2(self.fc2(x))
        x = self.a2(self.fc3(x))
        x = F.log_softmax(self.fc4(x), dim=1)

        return x


class Config(object):
    """parameters"""

    def __init__(self):
        self.device = 'cuda:3'
        # self.device = 'cpu'


if __name__ == "__main__":
    config = Config()
    print(config.device)

    trainset = FashionMNIST()
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=64, shuffle=True)

    # train CNN
    model = CNN(config).to(config.device)
    train_cnn_model(model, config)

    # train MLP
    # model = MLP(config).to(config.device)
    # train_linear_model(model, config)
