import torch.nn as nn


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=0, bias=False)


class BasicModle(nn.Module):

    def __init__(self):
        super(BasicModle, self).__init__()
        self.conv1 = conv3x3(1, 64)
        self.conv2 = conv3x3(64, 128)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = conv3x3(128, 256)
        self.conv4 = conv3x3(256, 128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv5 = conv3x3(128, 64)
        self.dropout= nn.Dropout(0.3)
        # self.fc1 = nn.Linear(2048, 1024)
        self.fc = nn.Linear(256, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool1(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.pool2(x)
        x = self.conv5(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
