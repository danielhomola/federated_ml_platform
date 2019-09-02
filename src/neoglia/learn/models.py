"""
A few example model architecture for learning with federated deep networks.
"""

import torch.nn as nn
import torch.nn.functional as F


class ConvNet(nn.Module):
    """
    Simple convolutional neural network for image data.

    Returns probabilities after softmax and not logits.
    """
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class FFNet(nn.Module):
    """
    Simple feed-forward network for tabular, numeric data with dropout.

    Returns probabilities after sigmoid and not logits.
    """
    def __init__(self, dense_size=256, dropout_rate=0.25, input_d=103):
        super(FFNet, self).__init__()
        self.fc1 = nn.Linear(input_d, dense_size)
        self.fc2 = nn.Linear(dense_size, int(dense_size/2))
        self.fc3 = nn.Linear(int(dense_size/2), int(dense_size/4))
        self.fc4 = nn.Linear(int(dense_size/4), 1)
        self.do = nn.Dropout(p=dropout_rate)
        self.sigm = nn.Sigmoid()

    def forward(self, x):
        x = self.do(F.relu(self.fc1(x)))
        x = self.do(F.relu(self.fc2(x)))
        x = self.do(F.relu(self.fc3(x)))
        return self.sigm(self.fc4(x))