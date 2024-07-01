

import torch.nn as nn

class DistortionClassifier(nn.Module):

    def __init__(self, out_classes=25):
        super(DistortionClassifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 64* 8, 128)
        self.fc2 = nn.Linear(128, out_classes)
        self.relu = nn.ReLU()



    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        #print(x.shape)
        x = x.view(-1, 64 * 64* 8)
        #print(x.shape)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

