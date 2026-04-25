import torch
import torch.nn as nn
import torch.nn.functional as F


class FedAvgNetMNIST(torch.nn.Module):
    def __init__(self, num_classes=10):
        super(FedAvgNetMNIST, self).__init__()

        self.conv2d_1 = nn.Conv2d(
            in_channels=1,
            out_channels=16,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(16)
        self.relu1 = nn.ReLU()
        self.max_pooling_1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2d_2 = nn.Conv2d(
            in_channels=16,
            out_channels=16,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(16)
        self.relu2 = nn.ReLU()
        self.max_pooling_2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.flatten = nn.Flatten()
        self.linear_1 = nn.Linear(7 * 7 * 16, 100)
        self.relu3 = nn.ReLU()

        self.linear_2 = nn.Linear(100, num_classes)

    def forward(self, x):
        if x.ndim < 4:
            x = torch.unsqueeze(x, 1)

        x = self.conv2d_1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.max_pooling_1(x)

        x = self.conv2d_2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.max_pooling_2(x)

        x = self.flatten(x)
        x = self.linear_1(x)
        x = self.relu3(x)
        x = self.linear_2(x)

        return x


class FedAvgNetCIFAR(torch.nn.Module):
    def __init__(self, num_classes=10):
        super(FedAvgNetCIFAR, self).__init__()
        self.conv2d_1 = torch.nn.Conv2d(3, 32, kernel_size=5, padding=2)
        self.max_pooling = nn.MaxPool2d(2, stride=2)
        self.conv2d_2 = torch.nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.flatten = nn.Flatten()
        self.linear_1 = nn.Linear(4096, 512)
        self.classifier = nn.Linear(512, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x, get_features=False):
        x = self.conv2d_1(x)
        x = self.relu(x)
        x = self.max_pooling(x)
        x = self.conv2d_2(x)
        x = self.relu(x)
        x = self.max_pooling(x)
        x = self.flatten(x)
        z = self.relu(self.linear_1(x))
        x = self.classifier(z)

        if get_features:
            return x, z
        else:
            return x


class FedAvgNetTiny(torch.nn.Module):
    def __init__(self, num_classes=10):
        super(FedAvgNetTiny, self).__init__()
        self.conv2d_1 = torch.nn.Conv2d(3, 32, kernel_size=5, padding=2)
        self.max_pooling = nn.MaxPool2d(2, stride=2)
        self.conv2d_2 = torch.nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.flatten = nn.Flatten()
        self.linear_1 = nn.Linear(16384, 512)
        self.classifier = nn.Linear(512, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x, get_features=False):
        x = self.conv2d_1(x)
        x = self.relu(x)
        x = self.max_pooling(x)
        x = self.conv2d_2(x)
        x = self.relu(x)
        x = self.max_pooling(x)
        x = self.flatten(x)
        z = self.relu(self.linear_1(x))
        x = self.classifier(z)

        if get_features:
            return x, z
        else:
            return x
